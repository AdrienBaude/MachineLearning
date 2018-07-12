# Reference https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

from keras.layers import Input, Dense, Lambda
from keras.models import Model
import keras.backend as K
from keras.datasets import mnist
from matplotlib import pyplot as plt
import os
import numpy as np
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_test = to_categorical(y_test, 10)

m = 50
n_z = 2
n_epoch = 10

inputs = Input(shape=(784,))
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0.)
    return mu + K.exp(log_sigma / 2) * eps

z = Lambda(sample_z)([mu, log_sigma])

decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

vae = Model(inputs, outputs)

encoder = Model(inputs, mu)

d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    return recon + kl

vae.compile(optimizer='adam', loss=vae_loss)


if os.path.isfile('generator.h5'):
    decoder.load_weights("generator.h5")
else:
    vae.fit(x_train, x_train, batch_size=m, nb_epoch=n_epoch)
    decoder.save_weights("generator.h5")

noise = np.random.normal(0, 1, (25, 2))
images = decoder.predict(noise)
images = 0.5 * images + 0.5

fig, axs = plt.subplots(5, 5)
cnt = 0
for i in range(5):
    for j in range(5):
        image = images[cnt].reshape((28, 28))
        axs[i, j].imshow(image, cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
plt.show()
