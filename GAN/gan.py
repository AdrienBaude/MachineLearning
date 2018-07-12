# Reference https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py

import os.path, numpy
from matplotlib import pyplot as plt
from keras.datasets import mnist

(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 127.5 - 1.
x_train = numpy.expand_dims(x_train, axis=3)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape, BatchNormalization, LeakyReLU
from keras.optimizers import Adam

optimizer = Adam(0.0002, 0.5)

generator = Sequential()
generator.add(Dense(256, input_shape=(100,)))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(512))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(1024))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape((28, 28, 1)))

discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28,28,1)))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(units=1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

discriminator.trainable = False

model = Sequential()
model.add(generator)
model.add(discriminator)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

if os.path.isfile('generator.h5'):
    generator.load_weights("generator.h5")
else:

    ones = numpy.ones((32, 1))
    zeros = numpy.zeros((32, 1))

    for epoch in range(30001):
        noise = numpy.random.normal(0, 1, (32, 100))
        fake = generator.predict(noise)
        true = x_train[numpy.random.randint(0, x_train.shape[0], 32)]

        d_loss_real = discriminator.train_on_batch(true, ones)
        d_loss_fake = discriminator.train_on_batch(fake, zeros)
        d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

        noise = numpy.random.normal(0, 1, (32, 100))
        g_loss = model.train_on_batch(noise, ones)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        if epoch % 3000 == 0:
            noise = numpy.random.normal(0, 1, (25, 100))
            images = generator.predict(noise)
            images = 0.5 * images + 0.5

            fig, axs = plt.subplots(5, 5)
            cnt = 0
            for i in range(5):
                for j in range(5):
                    axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images_%d.png" % epoch)
            plt.close()

    generator.save_weights("generator.h5")

noise = numpy.random.normal(0, 1, (25, 100))
images = generator.predict(noise)
images = 0.5 * images + 0.5

fig, axs = plt.subplots(5, 5)
cnt = 0
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
plt.show()
