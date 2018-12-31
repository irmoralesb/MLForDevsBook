from Chapter5.TransferFunction import TransferFunction
import numpy as np


class Linear(TransferFunction):
    def getTransferFunction(x):
        return x

    def getTransferFunctionDerivative(x):
        return np.ones(len(x))
