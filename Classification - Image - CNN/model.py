# Reference https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

# Imports --------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os.path

# Preprocessing --------------------------------------------------------------------------------------------------------

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
	validation_split=0.2)
	
train_generator = datagen.flow_from_directory(directory="./data", target_size=(32, 32),
    color_mode="rgb", class_mode="categorical", batch_size=32, subset="training")
test_generator = datagen.flow_from_directory(directory="./data", target_size=(32, 32),
    color_mode="rgb", class_mode="categorical", batch_size=32, subset="validation")

# Model ----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size, epochs=10,
                        validation_data=test_generator, validation_steps = test_generator.samples // test_generator.batch_size)
    model.save_weights("model.h5")

# Evaluation -----------------------------------------------------------------------------------------------------------

score = model.evaluate_generator(test_generator, test_generator.samples)
print('Accuracy: ', score[1])
