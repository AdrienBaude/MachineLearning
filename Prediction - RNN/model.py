# Reference https://www.superdatascience.com/deep-learning/

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os.path

start_time = time.time()

dataset_train = pd.read_csv('Google_Stock_Price.csv')
training_set = dataset_train.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1238):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

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
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

if os.path.isfile('model.h5'):
    classifier.load_weights("model.h5")
else:
    classifier.fit(X_train, y_train, epochs=100, batch_size=32)
    classifier.save_weights("model.h5")

print("\ntime elapsed: {:.2f}s".format(time.time() - start_time))
score = classifier.evaluate(X_train, y_train, verbose=0)
print('Accuracy: ', score[1])