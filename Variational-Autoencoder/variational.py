# Reference https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
import keras.backend as K
from keras.datasets import mnist
from matplotlib import pyplot as plt
import os
import numpy as np

(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(50, 2), mean=0.)
    return mu + K.exp(log_sigma / 2) * eps

inputs = Input(shape=(784,))
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(2, activation='linear')(h_q)
log_sigma = Dense(2, activation='linear')(h_q)
z = Lambda(sample_z)([mu, log_sigma])

decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

vae = Model(inputs, outputs)

encoder = Model(inputs, mu)

d_in = Input(shape=(2,))
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
    vae.fit(x_train, x_train, batch_size=50, nb_epoch=10)
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
