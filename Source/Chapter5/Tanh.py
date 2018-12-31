import numpy as np
from Chapter5.TransferFunction import TransferFunction


class Tanh(TransferFunction):  # Squash -1, 1
    def getTransferFunction(x):
        return np.tanh(x)

    def getTransferFunctionDerivative(x):
        return np.power(np.tanh(x), 2)
