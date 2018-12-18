import matplotlib.pyplot as plt
import numpy as np

# Also known as Gaussian function

mu = 0
sigma = 2
distro = np.random.normal(mu, sigma, 10000)
plt.hist(distro, 100, normed=True)
plt.show()