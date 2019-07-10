# Reference https://github.com/pranjal52/text_generators/blob/master/a_gigantic_model.ipynb

# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import np_utils
import os

# Preprocessing --------------------------------------------------------------------------------------------------------

text = (open(os.getcwd() + "/data/sonnets.txt", encoding="utf-8").read())
text = text.lower()

characters = sorted(list(set(text)))
n_to_char = {n: char for n, char in enumerate(characters)}
char_to_n = {char: n for n, char in enumerate(characters)}

X = []
Y = []
seq_length = 50
for i in range(0, len(text) - seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])
Y = np_utils.to_categorical(Y, num_classes=len(characters))

# Model ----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Embedding(len(characters), 10, input_length=seq_length))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(len(characters), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    model.fit(np.asarray(X), Y, epochs=50, batch_size=100)
    model.save_weights("model.h5")
	
# Evaluation -----------------------------------------------------------------------------------------------------------

start = "        the sun will set now and here and forever,"
string_mapped = [char_to_n[char] for char in start]
print(string_mapped)
full_string = [n_to_char[value] for value in string_mapped]
print(full_string)
for i in range(400):
    pred_index = model.predict_classes(np.asarray([string_mapped]), verbose=0)[0]
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt = ""
for char in full_string:
    txt = txt + char
print(txt)
