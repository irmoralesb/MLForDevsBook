import matplotlib.pyplot as plt
import numpy as np

def quadratic(var):
    return 2*pow(var, 2)


x = np.arange(0, 6, 1)
plt.plot(x, quadratic(x))
plt.plot([1, 4], [quadratic(1), quadratic(4)], linewidth=2.0)
plt.plot([1, 4], [quadratic(1), quadratic(1)], linewidth=3.0, label="Change in x")
plt.plot([4, 4], [quadratic(1), quadratic(4)], linewidth=3.0, label=" Change in y")
plt.legend()
plt.plot(x, 10*x - 8)
plt.plot()
plt.show()


## Next Step

initial_delta = 0.1
x1 = 1
for power in range(1,6):
    delta = pow(initial_delta, power)
    derivative_aprox= (quadratic(x1+delta)-quadratic(x1))/((x1+delta)-x1)
    print ("delta " + str(delta) + ", estimated derivative: " + str(derivative_aprox))


