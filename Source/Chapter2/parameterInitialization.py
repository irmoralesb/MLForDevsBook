import numpy as np

# A good approach is to initialize parameters with values different than 0
mu, sigma = 0, 1
dist = np.random.normal(mu, sigma, 1000)
print(dist)
