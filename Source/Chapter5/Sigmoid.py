import numpy as np

from Chapter5.TransferFunction import TransferFunction


class Sigmoid(TransferFunction):  # Squash 0,1
    def getTransferFunction(x):
        return 1 / (1 + np.exp(-x))

    def getTransferFunctionDerivative(x):
        return x * (1 - x)
