from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

y_true = [8,5,6,8,5,3,1,6,4,2,5,3,1,4]
y_pred = [8,5,6,8,5,2,3,4,4,5,5,7,2,6]
y = confusion_matrix(y_true, y_pred)
print(y)
plt.imshow(y, interpolation='nearest', cmap='plasma')
plt.xticks(np.arange(0, 8), np.arange(1, 9))
plt.yticks(np.arange(0, 8), np.arange(1, 9))
plt.show()

