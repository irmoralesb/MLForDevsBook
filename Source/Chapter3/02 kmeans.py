import numpy as np
import matplotlib
import matplotlib.pyplot as plt

samples = np.array([[1,2],[12,2],[0,1],[10,0],[9,1],[8,2],[0,10],[1,8],[2,9],[9,9],[10,8],[8,9] ], dtype=np.float)
centers = np.array([[3,2], [2,6], [9,3], [7,6]], dtype=np.float)
N = len(samples)

fig, ax = plt.subplots()
samples_trans = samples.transpose()
ax.scatter(samples_trans[0], samples_trans[1], marker='o', s=100)
centers_trans = centers.transpose()
ax.scatter(centers_trans[0], centers_trans[1], marker='s', s=100, color='black')
plt.plot()
plt.show()


def distance(sample, centroids):
    distances = np.zeros(len(centroids))
    for i in range(0, len(centroids)):
        dist = np.sqrt(sum(pow(np.subtract(sample, centroids[i]), 2)))
        distances[i] = dist
    return distances


def show_current_status(samples, centers, clusters, plotnumber):
    plt.subplot(620 + plotnumber)
    samples_transposed = samples.transpose()
    plt.scatter(samples_transposed[0], samples_transposed[1], marker='o', s=150, c=clusters)
    centers_transposed = centers.transpose()
    plt.scatter(centers_transposed[0], centers_transposed[1], marker='s', s=100, color='black')
    plt.plot()
    plt.show()


