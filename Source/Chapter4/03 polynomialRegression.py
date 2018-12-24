from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import seaborn.apionly as sns
import matplotlib.pyplot as plt

iris2 = sns.load_dataset('iris')

ix = iris2['petal_width']
iy = iris2['petal_length']

# Generate point used to represent the fitted function
x_plot = np.linspace(0, 2.6, 100)

# Create matrix versions of these arrays
X = ix[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.scatter(ix, iy, s=30, marker='o', label='training points')

for count, degree, in enumerate([3, 6, 20]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, iy)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label='degree %d' % degree)

plt.legend(loc='upper left')
plt.show()
# The coefficient = 20 adjust to the example data, but after that,the curve diverge, this is against the goal to be
# used with further data.
