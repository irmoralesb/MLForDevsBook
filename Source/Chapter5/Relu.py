from Chapter5.TransferFunction import TransferFunction


class Relu(TransferFunction):
    def getTransferFunction(x):
        return x * (x > 0)

    def getTransferFunctionDerivative(x):
        return 1 * (x > 0)
