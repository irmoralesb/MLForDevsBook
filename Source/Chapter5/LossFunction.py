import numpy as np


class LossFunction:
    def getLoss(y_, y):
        raise NotImplementedError


class L1(LossFunction):
    def getLoss(y_, y):
        return np.sum(y_ - y)


class L2(LossFunction):
    def getLoss(y_, y):
        return np.sum(np.power((y_ - y), 2))
