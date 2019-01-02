import numpy as np


class ConvolutionalOperation:
    def apply3x3kernel(self, image, kernel):  # Simple 3x3 kernel operation
        newimage = np.array(image)
        for m in range(1, image.shape[0] - 2):
            for n in range(1, image.shape[1] - 2):
                newelement = 0
                for i in range(0, 3):
                    for j in range(0, 3):
                        newelement = newelement + image[m - 1 + i][n - 1 + j] * kernel[i][j]
                newimage[m][n] = newelement
        return newimage
