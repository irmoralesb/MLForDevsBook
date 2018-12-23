import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

data, features = make_blobs(n_samples=100, n_features=2, centers=4, shuffle=True, cluster_std=0.8)
fig, ax = plt.subplots()
data_transposed = data.transpose()
ax.scatter(data_transposed[0], data_transposed[1], c=features, marker='o', s=100)
plt.plot()
plt.show()


def distance(sample, data):
    distances = np.zeros(len(data))
    for i in range(0, len(data)):
        dist = np.sqrt(sum(pow(np.subtract(sample, data[i]), 2)))
        distances[i] = dist
    return distances


def add_sample(new_sample, data, features):
    distances = np.zeros((len(data), len(data[0])))
    # calculates the distance between new sample and current data
    distances = distance(new_sample, data)
    closest_neighbors = np.argpartition(distances, 3)[:3]
    closest_groups = features[closest_neighbors]
    return np.argmax(np.bincount(closest_groups))


def knn(new_data, data, features):
    for i in new_data:
        test = add_sample(i, data, features)
        features = np.append(features, [test], axis=0)
        data = np.append(data, [i], axis=0)
    return data, features


new_samples = np.random.rand(20, 2) * 20 - 8
final_data, final_features = knn(new_samples, data, features)
fig, ax = plt.subplots()
ax.scatter(final_data.transpose()[0], final_data.transpose()[1], c=final_features, marker='o', s=100)
ax.scatter(new_samples.transpose()[0], new_samples.transpose()[1], c='none', marker='s', s=100)
plt.plot()
plt.show()
