import numpy as np
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

# Start seeing a general view of the data to try to determine what is the best approach

sns.pairplot(iris2, height=3.0)
plt.show()
# Based on the results, we chose
X = iris2['petal_width']
Y = iris2['petal_length']

plt.scatter(X, Y)


# plt.show()


# Creating the prediction function
def predict(alpha, beta, x_i):
    return beta * x_i + alpha


# Defining the error function
def error(alpha, beta, x_i, y_i):  # L1
    return y_i - predict(alpha, beta, x_i)


def sum_sq_e(alpha, beta, x, y):  # L2
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


# Now we define a function implementing the correlation method to find the parameters for our regression
def correlation_fit(x, y):
    beta = correlation(x, y) * np.std(y, ddof=1) / np.std(x, ddof=1)
    alpha = np.mean(y) - beta * np.mean(x)
    return alpha, beta


alpha, beta = correlation_fit(X, Y)
print("Alpha: {}".format(alpha))
print("Beta: {}".format(beta))

plt.scatter(X, Y)
xr = np.arange(0, 3.5)
plt.plot(xr, (xr * beta) + alpha)
plt.show()
