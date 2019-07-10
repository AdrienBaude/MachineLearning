# Reference https://blog.keras.io/building-autoencoders-in-keras.html

# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import os.path
import matplotlib.pyplot as plt
import cv2
import glob
from sklearn.model_selection import train_test_split

# Preprocessing --------------------------------------------------------------------------------------------------------

X = [cv2.imread(file) for file in glob.glob("./data/**/*.jpg")]
x_train, x_test = train_test_split(X, test_size=0.4)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train_noisy = x_train + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Model ----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(UpSampling2D())
model.add(Conv2D(3, 3, activation='sigmoid', padding='same'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    model.fit(x_train_noisy, x_train, epochs=10, batch_size=32)
    model.save_weights("model.h5")

# Evaluation -----------------------------------------------------------------------------------------------------------

imgs = model.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(imgs[i].reshape(28, 28, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
