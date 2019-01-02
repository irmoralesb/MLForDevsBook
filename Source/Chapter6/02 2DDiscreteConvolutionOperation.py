import matplotlib.pyplot as plt
import imageio
import numpy as np
import Chapter6.ConvolutionalOperation as co

arr = imageio.imread('blue_jay.bmp')[:, :, 0].astype(np.float)
plt.imshow(arr, cmap=plt.get_cmap('binary_r'))
plt.show()

kernels = {" Blur": [[1. / 16., 1. / 8., 1. / 16.], [1. / 8., 1. / 4., 1. / 8.], [1. / 16., 1. / 8., 1. / 16.]],
           " Identity": [[0, 0, 0], [0., 1., 0.], [0., 0., 0.]],
           " Laplacian": [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
           " Left Sobel": [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
           " Upper Sobel": [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]}

conv = co.ConvolutionalOperation()
plt.figure(figsize=(30, 30))
fig, axs = plt.subplots(figsize=(30, 30))
j = 1
print('It takes a while, please wait')
for key, value in kernels.items():
    axs = fig.add_subplot(3, 2, j)
    out = conv.apply3x3kernel(arr, value)
    plt.imshow(out, cmap=plt.get_cmap('binary_r'))
    j = j + 1

plt.show()
print('Completed')
