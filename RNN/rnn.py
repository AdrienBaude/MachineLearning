# Reference https://github.com/keras-team/keras/blob/master/examples/mnist_hierarchical_rnn.py

import time

start_time = time.time()

from keras.utils import np_utils
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10
img_rows, img_cols = 28, 28

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

classifier = Sequential()
classifier.add(SimpleRNN(units=50, input_shape=(28,28)))
classifier.add(Dense(10, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import os.path

if os.path.isfile('rnn.h5'):
    classifier.load_weights("rnn.h5")
else:
    classifier.fit(X_train, Y_train, batch_size=128, epochs=3, verbose=1)
    classifier.save_weights("rnn.h5")

print("time elapsed: {:.2f}s".format(time.time() - start_time))
score = classifier.evaluate(X_test, Y_test, verbose=0)
print('\nAccuracy: ', score[1])