# Reference https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

# Imports --------------------------------------------------------------------------------------------------------------

from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import keras.utils as ku
import numpy as np
import os, re

# Preprocessing --------------------------------------------------------------------------------------------------------

data = (open(os.getcwd() + "/data/sonnets.txt", encoding='utf-8').read())

tokenizer = Tokenizer(filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t')
seq_length = 50

data = data.replace("\n", " \n ")
data = data.replace(",", " , ")
data = data.replace(";", " ; ")
data = data.replace(":", " : ")
data = data.replace("!", " ! ")
data = data.replace("?", " ? ")
data = data.replace(".", " . ")

tokenizer.fit_on_texts([data])
data = tokenizer.texts_to_sequences([data])[0]
total_words = len(tokenizer.word_index) + 1

X = []
Y = []
for i in range(0, len(data) - seq_length, 1):
	token_list = data[i:i + seq_length + 1]
	X.append(token_list[:seq_length - 1])
	Y.append(token_list[seq_length - 1])
Y = ku.to_categorical(Y, num_classes=total_words)
X = np.asarray(X)
Y = np.asarray(Y)

# Model ----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Embedding(total_words, 10, input_length=seq_length - 1))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))    
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Training -------------------------------------------------------------------------------------------------------------

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    model.fit(X, Y, epochs=50, batch_size=100)
    model.save_weights("model.h5")

# Evaluation -----------------------------------------------------------------------------------------------------------

seed_text = """From fairest creatures we desire increase,
  That thereby beauty's rose might never die,
  But as the riper should by time decease,
  His tender heir might bear his memory:
  But thou, contracted to thine own bright eyes,
  Feed'st thy light's flame with self-substantial fuel,
  Making a famine where abundance lies,
  Thy self thy foe, to thy sweet self"""
seed_text = seed_text.replace("\n", " \n ")
seed_text = seed_text.replace(",", " , ")
seed_text = seed_text.replace(";", " ; ")
seed_text = seed_text.replace(":", " : ")
seed_text = seed_text.replace("!", " ! ")
seed_text = seed_text.replace("?", " ? ")
seed_text = seed_text.replace(".", " . ")
seed_text = seed_text.lower()
result = seed_text
seed_text = tokenizer.texts_to_sequences([seed_text])[0]

for _ in range(400):
	predicted = model.predict_classes(np.asarray([seed_text]), verbose=0)

	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break

	seed_text = seed_text[1:]
	seed_text.append(predicted[0])
	result += " " + output_word
print(result)
