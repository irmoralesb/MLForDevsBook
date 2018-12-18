import matplotlib.pyplot as plt
import numpy as np


plt.figure()
distro = np.random.binomial(1, .6, 10000) / 0.5
plt.hist(distro, 2, normed=1)
plt.show()


plt.figure()
distro = np.random.binomial(100, .6, 10000)/0.01
plt.hist(distro, 100, normed=1)
plt.show()
