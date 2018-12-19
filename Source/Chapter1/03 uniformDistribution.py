import matplotlib.pyplot as plt
import numpy as np

plt.figure()
uniform_low = 0.25
uniform_high = 0.8
uniform = np.random.uniform(uniform_low, uniform_high, 10000)
plt.hist(uniform, 50, density=1)
plt.show()
