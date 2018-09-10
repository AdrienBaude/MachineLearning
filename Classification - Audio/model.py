# Reference https://github.com/mtobeiyf/audio-classification

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import os.path
import time

start_time = time.time()

X = np.load('feat.npy')
y = np.load('label.npy').ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=193))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train - 1, num_classes=10)
y_test = keras.utils.to_categorical(y_test - 1, num_classes=10)

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    model.fit(X_train, y_train, epochs=1000, batch_size=64)
    model.save_weights("model.h5")

print("\ntime elapsed: {:.2f}s".format(time.time() - start_time))
score = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: ', score[1])