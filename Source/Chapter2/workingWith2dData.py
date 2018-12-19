import scipy.misc
from matplotlib import pyplot as plt
# Pillow library is also needed along scipy for images

testimg = scipy.misc.imread("data/blue_jay.jpg")
plt.imshow(testimg)
plt.show()

print("Image shape: {}".format(testimg.shape))

plt.subplot(131)
plt.imshow(testimg[:, :, 0], cmap="Reds")
plt.title("Red channel")
plt.subplot(132)
plt.imshow(testimg[:, :, 1], cmap="Greens")
plt.title("Green channel")
plt.subplot(133)
plt.imshow(testimg[:, :, 2], cmap="Blues")
plt.title("Blue channel")
plt.show()
