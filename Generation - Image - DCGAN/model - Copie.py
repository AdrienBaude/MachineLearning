# Reference https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

# Imports --------------------------------------------------------------------------------------------------------------

import os.path
import numpy
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Reshape, UpSampling2D, BatchNormalization
from keras.optimizers import Adam

# Preprocessing --------------------------------------------------------------------------------------------------------

(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train / 127.5 - 1.

# Model ----------------------------------------------------------------------------------------------------------------

optimizer = Adam(0.0002, 0.5)

generator = Sequential()
generator.add(Dense(7 * 7 * 128, activation="relu", input_shape=(100,)))
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D())
generator.add(Conv2D(128, 3, activation="relu", padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(UpSampling2D())
generator.add(Conv2D(64, 3, activation="relu", padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Conv2D(1, 3, activation="tanh", padding="same"))

discriminator = Sequential()
discriminator.add(Conv2D(64, 3, strides=2, activation="relu", input_shape=x_train.shape[1:]))
generator.add(BatchNormalization(momentum=0.8))
discriminator.add(Conv2D(128, 3, strides=2, activation="relu", input_shape=x_train.shape[1:]))
generator.add(BatchNormalization(momentum=0.8))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

discriminator.trainable = False
model = Sequential()
model.add(generator)
model.add(discriminator)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('model.h5'):
    generator.load_weights("model.h5")
else:

    ones = numpy.ones((100, 1))
    zeros = numpy.zeros((100, 1))

    for epoch in range(5000):
        fake = generator.predict(numpy.random.normal(0, 1, size=(100, 100)))
        true = x_train[numpy.random.randint(0, x_train.shape[0], 100)]

        d_loss_real = discriminator.train_on_batch(true, ones)
        d_loss_fake = discriminator.train_on_batch(fake, zeros)
        d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

        noise = numpy.random.normal(0, 1, size=(100, 100))
        g_loss = model.train_on_batch(noise, ones)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        if epoch % 1000 == 0:
            images = generator.predict(numpy.random.normal(0, 1, size=(25, 100)))
            images = 0.5 * images + 0.5
            fig, axs = plt.subplots(5, 5)
            cnt = 0
            for i in range(5):
                for j in range(5):
                    axs[i, j].imshow(images[cnt].reshape(28, 28), cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            plt.show()

    generator.save_weights("model.h5")

# Evaluation -----------------------------------------------------------------------------------------------------------

images = generator.predict(numpy.random.normal(0, 1, size=(25, 100)))
images = 0.5 * images + 0.5
fig, axs = plt.subplots(5, 5)
cnt = 0
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(images[cnt].reshape(28, 28), cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
plt.show()
