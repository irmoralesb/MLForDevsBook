import numpy as np
from sklearn import datasets
import seaborn.apionly as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", context="notebook")

iris2 = sns.load_dataset('iris')


def covariance(X, Y):
    xhat = np.mean(X)
    yhat = np.mean(Y)
    epsilon = 0
    for x, y in zip(X, Y):
        epsilon = epsilon + (x - xhat) * (y - yhat)
    return epsilon / (len(X) - 1)


# Testing results agains existing function
print("My covariance function: {}".format(covariance([1, 3, 4], [1, 0, 2])))
print("Numpy covariance function: {}".format(np.cov([1, 3, 4], [1, 0, 2])))


def correlation(X, Y):
    return (covariance(X, Y) / (np.std(X, ddof=1) * np.std(Y, ddof=1)))  # we had to indicat ddof=1 the unbiased std


print("My Correlation: {}".format(correlation([1, 1, 4, 3], [1, 0, 2, 2])))
print("Numpy corrcoef: {}".format(np.corrcoef([1, 1, 4, 3], [1, 0, 2, 2])))
