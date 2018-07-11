# Reference https://www.superdatascience.com/deep-learning/

import time

start_time = time.time()

import numpy as np
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price.csv')
training_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1238):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

classifier = Sequential()
classifier.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50, return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50, return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=50))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1))
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape'])

import os.path

if os.path.isfile('rnn.h5'):
    classifier.load_weights("rnn.h5")
else:
    classifier.fit(X_train, y_train, epochs=100, batch_size=32)
    classifier.save_weights("rnn.h5")

dataset = dataset_train['Open']
inputs = dataset[len(dataset) - 80:].values
inputs = inputs.reshape(-1, 1)
y_test = inputs[60:80]
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = classifier.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

mape = 0
for i in range(0,20):
    mape+=abs((y_test[i][0]-predicted_stock_price[i][0])/y_test[i][0])
mape*=5

print("time elapsed: {:.2f}s".format(time.time() - start_time))
print("\nAccuracy: %.2f%%" % (100-mape))