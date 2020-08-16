import numpy as np
import cv2
import gc


class Image(object):
    """The base-class for image input"""

    def __init__(self, image=None):
        if image is not None:
            self.image = image

    def imread(self, path):
        self.image = cv2.imread(path)

    def extract_convWin(self, x, y, offset, xOff=None, yOff=None):
        return {"win": self.image[x:x + offset, y:y + offset], "i": x, "j": y, "x": xOff, "y": yOff}

    def save_image(self, path):
        cv2.imwrite(path, self.image)


class ImageGen(Image):
    """Utility functions for pre-processing the images"""

    def __init__(self, image=None):
        super(ImageGen, self).__init__(image)
        self.delta = None

    def add_noise(self, delta):
        self.delta = delta
        noise = np.random.rand(self.image.shape[0], self.image.shape[1])
        noise[noise > delta] = 1
        noise[noise <= delta] = 0
        self.image = self.image * noise


class convWins_Generator(Image):
    """A class for getting a "generator" of possible convolution windows"""

    def __init__(self, image=None):
        super(convWins_Generator, self).__init__(image)

    def extract_jumpingWins(self, offset):
        iRange = [offset * x for x in range(self.image.shape[0] // offset)]
        jRange = [offset * x for x in range(self.image.shape[1] // offset)]

        for i in iRange:
            for j in jRange:
                yield self.extract_convWin(i, j, offset, iRange.index(i), jRange.index(j))

    def extract_slidingWins(self, offset):
        iRange = [x for x in range(self.image.shape[0]) if x + offset - 1 < self.image.shape[0]]
        jRange = [x for x in range(self.image.shape[1]) if x + offset - 1 < self.image.shape[1]]

        for i in iRange:
            for j in jRange:
                yield self.extract_convWin(i, j, offset)


class Conv(convWins_Generator, ImageGen):
    """The operator class for different Convloutions.

    params:

    name --> name of the image or operation.

    image --> a numpy array of image.

    functions:

    conv_kernals --> convolution of a specific kernal(or pre-loaded kernals, refer doc-string of the fuction) on to the image.

    conv_pooling --> pooling on the image (for more info refer its doc-string).

    gaussian_conv --> gaussian processing on the image(for more info refer its doc-string). 

    get_gaussian --> returns a gaussian kernal of desired size and standard deviation.    

    get_sobel --> returns the complete sobel edges of the image.

    single_threshold --> thresholding the image.

    double_threshold --> thresholding the image agains two limits.

    hysteresis --> suprressing the edges-points which are not connected to strong edges-points in a double thresholded image. 

    save_output --> saves the output of the recent operation. 
    """

    def __init__(self, name, image=None):
        super(Conv, self).__init__(image)
        self.kernals = {
            "sobelX": {
                "kernal": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                "offset": 0.125
            },
            "sobelY": {
                "kernal": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                "offset": 0.125
            },
            "gaussian": {
                "kernal": self.get_gaussian(3, 1),
                "offset": 1
            },
            "laplacian": {
                "kernal": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
                "offset": 1
            }
        }
        self.name = name

        self.pooling = {
            "average": np.average,
            "median": np.median,
            "max": np.max,
            "min": np.min
        }

    def conv_kernals(self, mode, kernalType, Ckernal=None, truConv=True, save=False):
        """
        params:

        mode --> "sliding" or "jumping" convolutions.

        kernalType --> name of the kernal.
                       the pre loaded kernals are - "sobelX", "sobelY" and "gaussian 3x3" 

        Ckernal --> dictionary with a numpy array(square matrix) "kernal" and a "offset" for normalization,
                    if used the pre loaded kernals this param can be ignored if not "keranlType" still should be specified.

        save --> True if the output is to be saved to disk.

        retuns: 
        The resultant numpy array image.
        """

        if Ckernal is None:
            kernal = self.kernals[kernalType]["kernal"]
            offset = self.kernals[kernalType]["offset"]

        else:
            kernal = Ckernal["kernal"]
            offset = Ckernal["offset"]

        h, w = self.image.shape[0], self.image.shape[1]
        if truConv:
            h, w = self.image.shape[0] - kernal.shape[0] + 1, self.image.shape[1] - kernal.shape[0] + 1

        if mode == "sliding":

            delta = self.image.shape[0] - kernal.shape[0]
            delta = delta % kernal.shape[0]

            slide_ConvOutput = np.zeros([h, w])
            for patch in self.extract_slidingWins(kernal.shape[0]):
                slide_ConvOutput[patch["i"], patch["j"]] = offset * np.sum(patch["win"] * kernal)
            self.output = slide_ConvOutput

        else:
            jump_ConvOutput = np.zeros([self.image.shape[0] // kernal.shape[0],
                                        self.image.shape[1] // kernal.shape[0]])
            for patch in self.extract_jumpingWins(kernal.shape[0]):
                jump_ConvOutput[patch["x"], patch["y"]] = offset * np.sum(patch["win"] * kernal)
            self.output = jump_ConvOutput

        if save:
            cv2.imwrite(f"{self.name}_{kernalType}_with_{mode}_conv.png", self.output)
        gc.collect()

        return self.output

    def conv_pooling(self, pooling_type, mode, winSize, truConv=True, save=False):
        """
        params:

        pooling_type --> "average", "median", "max" and "min"

        mode --> sliding or jumping convolutions

        winSize --> window size.

        save --> True if the output is to be saved to disk.

        returns : the resultant numpy array image.
        """

        func = self.pooling[pooling_type]
        offset = winSize

        h, w = self.image.shape[0], self.image.shape[1]
        if truConv:
            h, w = self.image.shape[0] - offset + 1, self.image.shape[1] - offset + 1

        if mode == "sliding":
            slide_poolingOut = np.zeros([h, w])
            for patch in self.extract_slidingWins(offset):
                slide_poolingOut[patch["i"], patch["j"]] = func(patch["win"])
            self.output = slide_poolingOut

        else:
            jump_poolingOut = np.zeros([self.image.shape[0] // offset, self.image.shape[1] // offset])
            for patch in self.extract_jumpingWins(offset):
                jump_poolingOut[patch["x"], patch["y"]] = func(patch["win"])
            self.output = jump_poolingOut

        if save:
            cv2.imwrite(f"{self.name}_averageSmoothing_offset_{offset}_with_{mode}_conv.png", self.output)
        gc.collect()

        return self.output

    def gaussian_conv(self, mode, size, alpha, save=False):
        """
        Gaussian smoothing of a image.

        params:

        mode: "sliding" or "jumping"

        size: size of the gaussian window

        alpha: standard deviation.

        save: True to save the result to disk

        returns:

        the output of the convolution.
        """
        Ckernal = {
            "kernal": self.get_gaussian(size, alpha),
            "offset": 1
        }

        self.output = self.conv_kernals(mode, f"gaussianKernal_Size_{size}_alpha_{alpha}", Ckernal, save)
        return self.output

    def get_gaussian(self, size, alpha):
        """
        Get a gaussian kernal of desired size.

        params:

        size: size of the kernal

        alpha: standard deviation

        returns:

        a numpy array gaussian kernal with the desired size and standard deviation.
        """
        G = np.zeros([size, size])
        k = (size - 1) / 2
        nTerm = 1 / (2 * np.math.pi * np.square(alpha))
        for i in range(1, size + 1):
            for j in range(1, size + 1):
                x = i - k - 1
                y = j - k - 1
                G[i - 1, j - 1] = nTerm * np.exp(-((np.square(x) + np.square(y)) / (2 * np.square(alpha))))
        if np.sum(G) != 0:
            G /= G.sum()
        else:
            raise Exception("Zero divide error")

        return G

    def get_sobel(self, mode, save=False):
        """
        For computing the complete sobel edges.

        params:

        mode: "sliding" or "jumping"

        save: True to save the result

        returns:

        the output of the operation.
        """
        sobelX = self.conv_kernals(mode, "sobelX")
        sobelY = self.conv_kernals(mode, "sobelY")
        self.output = np.sqrt(np.square(sobelX) + np.square(sobelY))
        gc.collect()

        if save:
            cv2.imwrite(f"{self.name}_SobelXY_with_{mode}_conv.png", self.output)
        return self.output

    def __threshold(self, thres, value, mode):
        """
        A utility function for "n" thresholding the image.

        params:

        thres: threshold value.

        value: value to asign for the operation

        mode: "min" or "max"

        returns:

        the thresholded image.

        """
        if mode == "max":
            self.output[self.output >= thres] = value
        else:
            self.output[self.output <= thres] = value

        return self.output

    def single_threshold(self, thres, max_value, min_value, save=False):
        """
        single thresholding the image.

        params:

        threshold: single threshold value.

        max_value: value to assign, if above threshold.

        min value: value to assign, if below threshold.

        returns:

        the single thresholded image.
        """
        self.output = self.image.copy()
        gc.collect()
        self.__threshold(thres, max_value, "max")
        self.__threshold(thres, min_value, "min")
        gc.collect()

        if save:
            self.save_output(f"{self.name}_threshold_{thres}.png")

        return self.output

    def double_threshold(self, max_thres, max_value, min_thres, min_value, save=False):
        """
        double thresholding the image.

        params:

        max_thres: upper threshold value.

        min_thres: lower threshold value.        

        max_value: value to assign, if above threshold.

        min value: value to assign, if below threshold.

        returns:

        Applies double thresholding on the image and returns.
        """
        self.output = self.image.copy()
        self.__threshold(max_thres, max_value, "max")
        self.__threshold(min_thres, min_value, "min")
        gc.collect()

        if save:
            self.save_output(f"{self.name}_doubleThreshold_{max_thres}_{min_thres}.png")

        return self.output

    def hysteresis(self, winSize, max_thres, min_thres, max_value, min_value, save=False):
        """
        single thresholding the image.

        params:

        winSize: window size.

        max_thres: upper threshold value.

        min_thres: lower threshold value.        

        max_value: value to assign, if above threshold.

        min value: value to assign, if below threshold.

        returns:

        Applies hysteresis on the image and returns it.
        """

        temp = self.double_threshold(max_thres, max_value, min_thres, min_value).copy()
        imageTemp = self.image.copy()
        self.image = self.output.copy()

        self.conv_pooling("max", "sliding", winSize, truConv=False)

        prime_mask = np.logical_and(temp >= min_thres, temp <= max_thres)
        mask_u = np.logical_and(prime_mask, self.output >= max_value)
        mask_d = np.logical_and(prime_mask, self.output < max_value)

        temp[mask_u] = max_value
        temp[mask_d] = min_value

        self.image = imageTemp
        self.output = temp
        gc.collect()

        if save:
            self.save_output(f"{self.name}_hysteresis_{max_thres}_{min_thres}_winSize_{winSize}.png")

        return self.output

    def save_output(self, path):
        """Saves the output of the recent operation

        Params:

        path : path to save the file
        """
        cv2.imwrite(path, self.output)
