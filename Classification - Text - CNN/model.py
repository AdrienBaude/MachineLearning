# Reference https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn

# Imports --------------------------------------------------------------------------------------------------------------

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from keras.models import Sequential
from keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, Flatten, GlobalMaxPooling1D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sklearn.datasets as skds
from pathlib import Path
import os

# Preprocessing --------------------------------------------------------------------------------------------------------

if os.path.isfile('./data/dataframe.pickle'):
    data = pd.read_pickle('./data/dataframe.pickle')
else:	
	files = skds.load_files(os.getcwd() + "\\data", load_content=False)

	label_index = files.target
	label_names = files.target_names
	labelled_files = files.filenames

	data_tags = ["filename", "category", "news"]
	data_list = []

	i = 0
	for f in labelled_files:
		data_list.append((f, label_names[label_index[i]], Path(f).read_text()))
		i += 1

	data = pd.DataFrame.from_records(data_list, columns=data_tags)
	data.to_pickle('./data/dataframe.pickle')

encoder = LabelBinarizer()
encoder.fit(data["category"])
Y = encoder.transform(data["category"])

if os.path.isfile('./data/tokenizer.pickle'):
    tokenizer = Tokenizer()
    with open('./data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data["news"])
    with open('./data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
X = tokenizer.texts_to_sequences(data["news"])
X = pad_sequences(X, padding='post', maxlen=500)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Model ----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 50, input_length=500))
model.add(Conv1D(128, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    model.save_weights("model.h5")

# Evaluation -----------------------------------------------------------------------------------------------------------

score = model.evaluate(x_test, y_test)
print('Accuracy: ', score[1])