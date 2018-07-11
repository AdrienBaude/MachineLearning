# Reference https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

import time

start_time = time.time()

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import os.path

if os.path.isfile('ann.h5'):
    model.load_weights("ann.h5")
else:
    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))
    model.save_weights("ann.h5")

print("time elapsed: {:.2f}s".format(time.time() - start_time))
score = model.evaluate(x_test, y_test, verbose=0)
print('\nAccuracy: ', score[1])
