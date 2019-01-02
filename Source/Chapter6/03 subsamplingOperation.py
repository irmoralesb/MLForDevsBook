import numpy as np
import matplotlib.pyplot as plt
import Chapter6.PoolingOperation as po
import imageio

arr = imageio.imread('blue_jay.bmp')[:, :, 0].astype(np.float)
plt.figure(figsize=(30, 30))
pool = po.PoolingOperation()
fig, axs = plt.subplots(figsize=(20, 10))
axs = fig.add_subplot(1, 2, 1)
plt.imshow(arr, cmap=plt.get_cmap('binary_r'))
out = pool.apply2x2pooling(arr, 1)
axs = fig.add_subplot(1, 2, 2)
plt.imshow(out, cmap=plt.get_cmap('binary_r'))
plt.show()
