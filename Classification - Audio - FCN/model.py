# Reference http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

# Imports --------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Preprocessing --------------------------------------------------------------------------------------------------------

with open("./data/X.npy", 'rb') as file:
    X = np.load(file)
with open("./data/Y.npy", 'rb') as file:
    Y = np.load(file)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model ----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))
    model.save_weights("model.h5")

# Evaluation -----------------------------------------------------------------------------------------------------------

score = model.evaluate(x_test, y_test)
print('Accuracy: ', score[1])
