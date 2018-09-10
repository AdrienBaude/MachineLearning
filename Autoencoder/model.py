# Reference https://blog.keras.io/building-autoencoders-in-keras.html

from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import os.path
import matplotlib.pyplot as plt

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoder = Sequential()
encoder.add(Dense(128, activation='relu', input_shape=(784,)))
encoder.add(Dense(64, activation='relu'))
encoder.add(Dense(32, activation='relu'))

decoder = Sequential()
decoder.add(Dense(64, activation='relu', input_shape=(32,)))
decoder.add(Dense(128, activation='relu'))
decoder.add(Dense(784, activation='sigmoid'))

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)
autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if os.path.isfile('encoder.h5') and os.path.isfile('decoder.h5') and os.path.isfile('model.h5'):
    encoder.load_weights("encoder.h5")
    decoder.load_weights("decoder.h5")
    autoencoder.load_weights("model.h5")
else:
    autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, shuffle=True)
    encoder.save_weights("encoder.h5")
    decoder.save_weights("decoder.h5")
    autoencoder.save_weights("model.h5")

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
