import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.pyplot import plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout

df = pd.read_csv('date/elec_load.csv', error_bad_lines=False)
plt.subplot()
plot_test, = plt.plot(df.values[:1500], label='Load')
plt.legend(handles=[plot_test])

print(df.describe())
array = (df.values - 145.33) / 338.21  # minus the mean and divided between the max value
plt.subplot()
plot_test, = plt.plot(array[:1500], label='Normalized Load')
plt.legend(handles=[plot_test])

listX = []
listY = []
x = {}
y = {}

for i in range(0, len(array) - 6):
    listX.append(array[i:i + 5].reshape([5, 1]))
    listY.append(array[i + 6])

arrayX = np.array(listX)
arrayY = np.array(listY)

x['train'] = arrayX[0:13000]
x['test'] = arrayY[13000:14000]
y['train'] = arrayY[0:13000]
y['test'] = arrayY[13000, 14000]

# Build the model
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, 1), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=200, input_shape=(None, 100), return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")

# Fit the model to the data
model.fit(x['train'], y['train'], batch_size=512, epochs=10, validation_split=0.08)

# Rescale the test dataset and predicted data
test_results = model.predict(x[' test'])
test_results = test_results * 338.21 + 145.33
y[' test'] = y[' test'] * 338.21 + 145.33
plt.figure(figsize=(10, 15))
plot_predicted, = plt.plot(test_results, label=' predicted')
plot_test, = plt.plot(y[' test'], label=' test')
plt.legend(handles=[plot_predicted, plot_test])
