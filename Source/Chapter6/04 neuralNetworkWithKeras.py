from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import cv2

model = InceptionV3()
model.compile(optimizer=SGD(), loss='categorical_crossentropy')

# resize into VGG16 trained images' format
im = cv2.resize(cv2.imread('blue_jay.jpg'), (299, 299))
im = np.expand_dims(im, axis=0)
im = im / 255
im = im * 2
plt.figure(im[0], cmap=plt.get_cmap('binary_r'))
plt.show()

out = model.predict(im)
print(' Predicted:', decode_predictions(out, top=3)[0])
print(np.argmax(out))
