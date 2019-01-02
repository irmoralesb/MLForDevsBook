import numpy as np


class PoolingOperation:
    def apply2x2pooling(self, image, stride):  # Simple 2x2 kernel operation
        newimage = np.zeros((int(image.shape[0] / 2), int(image.shape[1] / 2)), np.float32)
        for m in range(1, image.shape[0] - 2, 2):
            for n in range(1, image.shape[1] - 2, 2):
                newimage[int(m / 2), int(n / 2)] = np.max(image[m:m + 2, n:n + 2])
        return newimage
