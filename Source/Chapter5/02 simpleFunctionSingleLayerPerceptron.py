import numpy as np
import matplotlib.pyplot as plt
import Chapter5.Sigmoid as sigmoid
import Chapter5.Tanh as tanh
import Chapter5.Relu as relu
import Chapter5.Linear as linear
import Chapter5.LossFunction as lf

# from pprint import pprint
# from sklearn import datasets
plt.style.use('fivethirtyeight')


def graphTransferFunction(function):
    x = np.arange(-2.0, 2.0, 0.01)
    plt.figure(figsize=(18, 8))
    ax = plt.subplot(121)
    ax.set_title(function.__name__)
    plt.plot(x, function.getTransferFunction(x))
    ax = plt.subplot(122)
    ax.set_title('Derivative of ' + function.__name__)
    plt.plot(x, function.getTransferFunctionDerivative(x))
    plt.show()


graphTransferFunction(sigmoid.Sigmoid)

ws = np.arange(-1.0, 1.0, 0.2)
bs = np.arange(-2.0, 2.0, 0.2)
xs = np.arange(-4.0, 4.0, 0.1)
plt.figure(figsize=(20, 10))
ax = plt.subplot(121)
for i in ws:
    plt.plot(xs, sigmoid.Sigmoid.getTransferFunction(i * xs), label=str(i))
ax.set_title('Sigmoid variants in w')
plt.legend(loc='upper left')

ax = plt.subplot(122)
for i in bs:
    plt.plot(xs, sigmoid.Sigmoid.getTransferFunction(i + xs), label=str(i))
ax.set_title('Sigmoid variants in b')
plt.legend(loc='upper left')
plt.show()

graphTransferFunction(tanh.Tanh)

graphTransferFunction(relu.Relu)

graphTransferFunction(linear.Linear)

# L1 vs L2 loss functions
sampley_ = np.array([.1, .2, .3, -.4, -1, -3, 6, 3])
sampley = np.array([.2, -.2, .6, .10, 2, -1, 3, -1])

ax.set_title('Sigmoid variants in b')
plt.figure(figsize=(10, 10))
ax = plt.subplot()
plt.plot(sampley_ - sampley, label='L1')
plt.plot(np.power((sampley_ - sampley), 2), label="L2")
ax.set_title('L1 vs L2 initial comparison')
plt.legend(loc='best')
plt.show()

# input dataset
X = np.array([[0, 1, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]])

# initialize weights with randomly with mean 0
W = 2 * np.random.random((3, 1)) - 1
print(W)

errorlist=np.empty(3)
weighthistory=np.array(0)
resultshistory=np.array(0)

for iter in range(100):
    # forward propagation
    l0 = X
    l1 = sigmoid.Sigmoid.getTransferFunction(np.dot(l0, W))
    resultshistory = np.append(resultshistory, l1)

    # Error calculation
    l1_error = y - l1
    errorlist = np.append(errorlist, l1_error)

    # Back propagation 1: Get the deltas
    l1_delta = l1_error * sigmoid.Sigmoid.getTransferFunctionDerivative(l1)

    # update weights
    W = W + np.dot(l0.T, l1_delta)
    weighthistory = np.append(weighthistory, W)

print(l1)

# To better understand the process, let's have a look at how the parameters change over time. First, let's graph the
# neuron weights. As you can see, they go from a random state to accepting the whole values of the first column
# (which is always right), going to almost 0 for the second column (which is right 50% of the time),
# and then going to -2 for the third (mainly because it has to trigger 0 in the first two elements of the table):

plt.figure(figsize=(20, 20))
print(W)
plt.imshow(np.reshape(weighthistory[1:], (-1, 3))[:40], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# Let's also review how our solutions evolved (during the first 40 iterations) until we reached the last iteration;
# we can clearly see the convergence to the ideal values:

plt.figure(figsize=(20, 20))
plt.imshow(np.reshape(resultshistory[1:], (-1, 4))[:40], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# We can see how the error evolves and tends to be zero through the different epochs. In this case, we can observe
# that it swings from negative to positive, which is possible because we first used an L1 error function:
plt.figure(figsize=(10, 10))
plt.plot(errorlist)
plt.show()
