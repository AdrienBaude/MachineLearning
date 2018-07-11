# Reference https://www.superdatascience.com/deep-learning/

import time

start_time = time.time()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32,
                                            class_mode='binary')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import os.path

if os.path.isfile('cnn.h5'):
    classifier.load_weights("cnn.h5")
else:
    classifier.fit_generator(training_set, epochs=25)
    classifier.save_weights("cnn.h5")

print("time elapsed: {:.2f}s".format(time.time() - start_time))
score, acc = classifier.evaluate_generator(test_set, steps=32)
print("\nAccuracy: %.2f%%" % (acc * 100))
