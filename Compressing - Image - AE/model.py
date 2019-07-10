# Reference https://blog.keras.io/building-autoencoders-in-keras.html

# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
import glob
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dense
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import pickle

# Preprocessing --------------------------------------------------------------------------------------------------------
	
X = [cv2.imread(file) for file in glob.glob("./data/**/*.jpg")]
x_train, x_test = train_test_split(X, test_size=0.4)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Model ----------------------------------------------------------------------------------------------------------------

encoder = Sequential()
encoder.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=x_train.shape[1:]))
encoder.add(MaxPooling2D())
encoder.add(Flatten())
encoder.add(Dense(32, activation='relu'))
encoder.summary()

decoder = Sequential()
decoder.add(Dense(6272, activation='relu', input_shape=(32,)))
decoder.add(Reshape((14, 14, 32)))
decoder.add(Conv2D(32, 3, activation='relu', padding='same'))
decoder.add(UpSampling2D())
decoder.add(Conv2D(3, 3, activation='sigmoid', padding='same'))
decoder.summary()

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
autoencoder.summary()

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('encoder.h5') and os.path.isfile('decoder.h5'):
    encoder.load_weights("encoder.h5")
    decoder.load_weights("decoder.h5")
else:
    autoencoder.fit(x_train, x_train, epochs=10, batch_size=32)
    encoder.save_weights("encoder.h5")
    decoder.save_weights("decoder.h5")

# Evaluation -----------------------------------------------------------------------------------------------------------

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
