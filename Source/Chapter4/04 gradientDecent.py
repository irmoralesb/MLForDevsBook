import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid', context='notebook')


def least_squares(b0, b1, points):
    total_error = 0
    n = float(len(points))
    for x, y in points:
        total_error += (y - (b1 * x + b0)) ** 2
    return total_error / 2. * n


def step_gradient(b0_current, b1_current, points, learningRate):
    b0_gradient = 0
    b1_gradient = 1
    n = float(len(points))
    for x, y in points:
        b0_gradient += (1 / n) * (y - ((b1_current * x) + b0_current))
        b1_gradient += (1 / n) * x * (y - ((b1_current * x) + b0_current))
    new_b0 = b0_current + (learningRate * b0_gradient)
    new_b1 = b1_current + (learningRate * b1_gradient)
    return [new_b0, new_b1, least_squares(new_b0, new_b1, points)]


def run_gradient_descent(points, starting_b0, starting_b1, learning_rate, num_iterations):
    b0 = starting_b0
    b1 = starting_b1
    slope = []
    intersect = []
    error = []
    for i in range(num_iterations):
        b0, b1, e = step_gradient(b0, b1, np.array(points), learning_rate)
        slope.append(b1)
        intersect.append(b0)
        error.append(e)
    return [b0, b1, e, slope, intersect, error]


# This process could prove inefficient when the convergence rate is high, wasting CPU iterations.
# A more clever stop condition would consist of adding an acceptable error value, which would stop the iteration.


iris = sns.load_dataset('iris')
X = iris['petal_width'].tolist()
Y = iris['petal_length'].tolist()
points = np.dstack((X, Y))[0]

learning_rate = 0.0001
initial_b0 = 0
initial_b1 = 0
num_iterations = 1000
[b0, b1, e, slope, intersect, error] = run_gradient_descent(points, initial_b0, initial_b1, learning_rate,
                                                            num_iterations)
plt.figure(figsize=(7, 5))
plt.scatter(X, Y)
xr = np.arange(0, 3.5)
plt.plot(xr, (xr * b1) + b0)
plt.title('Regression, alpha= 0.001, initial values=(0,0), it=1000')
plt.show()
# We are far from what we are looking for, so take a look at errors

plt.figure(figsize=(7, 5))
xr = np.arange(0, 1000)
plt.plot(xr, np.array(error).transpose())
plt.title("Error for 1000 iterations")
plt.show()
# the process seems to be working, but a little slow, we can try to increase the step by a factor of 10 to see if
# it converges quickly

learning_rate = 0.001  # This is the updated value
initial_b0 = 0
initial_b1 = 0
num_iterations = 1000
[b0, b1, e, slope, intersect, error] = run_gradient_descent(points, initial_b0, initial_b1, learning_rate,
                                                            num_iterations)
plt.figure(figsize=(7, 5))
xr = np.arange(0, 1000)
plt.plot(xr, np.array(error).transpose())
plt.title("Error for 1000 iterations, increased step by tenfold")
plt.show()
# This is more alike to what we are looking

plt.figure(figsize=(7, 5))
plt.scatter(X, Y)
xr = np.arange(0, 3.5)
plt.plot(xr, (xr * b1) + b0)
plt.title("Regression, alpha= 0.01, initial values = (0,0), it=1000")
plt.show()

# To go faster we my try this:
# but this end up to a bad move, due to error now tends to infinity
learning_rate = 0.85  # last one was 0.0001
initial_b0 = 0
initial_b1 = 0
num_iterations = 1000
[b0, b1, e, slope, intersect, error] = run_gradient_descent(points, initial_b0, initial_b1, learning_rate,
                                                            num_iterations)
plt.figure(figsize=(7, 5))
xr = np.arange(0, 1000)
plt.plot(xr, np.array(error).transpose())
plt.title('Error for 1000 iterations, big step')

# No trying with semi-random initial parameters
learning_rate = 0.001  # Same as last time
initial_b0 = 0.8  # pseudo random value
initial_b1 = 1.5  # pseudo random value
num_iterations = 1000
[b0, b1, e, slope, intersect, error] = run_gradient_descent(points, initial_b0, initial_b1, learning_rate,
                                                            num_iterations)
plt.figure(figsize=(7, 5))
xr = np.arange(0, 1000)
plt.plot(xr, np.array(error).transpose())
plt.title('Error for 1000 iterations, step 0.001, random initial parameter values')
plt.show()
# As you can see, even if you have the same sloppy error rate, the initial error value decreases tenfold
# (from 2e5 to 2e4). Now let's try a final technique to improve the convergence of the parameters based on the
# normalization of the input values.

learning_rate = 0.001  # Same as last time
initial_b0 = 0.8  # pseudo random value
initial_b1 = 1.5  # pseudo random value
num_iterations = 1000
x_mean = np.mean(points[:, 0])
y_mean = np.mean(points[:, 1])
x_std = np.std(points[:, 0])
y_std = np.std(points[:, 1])

X_normalized = (points[:, 0] - x_mean) / x_std
Y_normalized = (points[:, 1] - y_mean) / y_std

plt.figure(figsize=(7, 5))
plt.scatter(X_normalized, Y_normalized)
plt.show()
# Now that we have this set of clean and tidy data, let's try again with the last slow convergence parameters,
# and see what happens to the error minimization speed:

points = np.dstack((X_normalized, Y_normalized))[0]
learning_rate = 0.001  # Same as last time
initial_b0 = 0.8  # pseudo random value
initial_b1 = 1.5  # pseudo random value
num_iterations = 1000
[b0, b1, e, slope, intersect, error] = run_gradient_descent(points, initial_b0, initial_b1, learning_rate,
                                                            num_iterations)
plt.figure(figsize=(7, 5))
xr = np.arange(0, 1000)
plt.plot(xr, np.array(error).transpose())
plt.title('Error for 1000 iterations, step 0.001, random initial parameter values, normalized initial values')
plt.show()
# A very good starting point indeed! Just by normalizing the data, we have half the initial error values, and the error
# went down 20% after 1,000 iterations. The only thing we have to remember is to denormalize after we have the results,
# in order to have the initial scale and data center. So, that's all for now on gradient descent. We will be revisiting
# it in the next chapters for new challenges.
