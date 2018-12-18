import matplotlib.pyplot as  plt
import numpy as np

# Similar to the normal distribution, but with the morphological difference of having
# a more elongated tail.
# The main importance of this distribution lies in its cumulative distribution function (CDF)

mu = 0.5
sigma = 0.5
distro2 = np.random.logistic(mu, sigma, 10000)
plt.hist(distro2, 50, normed=True)
distro = np.random.normal(mu, sigma, 10000)
plt.hist(distro, 50, normed=True)
plt.show()

# sigmoid curve
plt.figure()
logistic_cumulative = np.random.logistic(mu, sigma, 10000)/0.02
plt.hist(logistic_cumulative, 50, density=1, cumulative=True)
plt.show()