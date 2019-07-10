# Reference https://www.opencodez.com/python/text-classification-using-keras.htm

# Imports --------------------------------------------------------------------------------------------------------------

import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sklearn.datasets as skds
from pathlib import Path
import os
import nltk
import re
import string

# Cleaner --------------------------------------------------------------------------------------------------------------

def clean(text):
    text = text.translate(string.punctuation)
    text = text.lower().split()
    stops = set(nltk.corpus.stopwords.words('english'))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

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
	data['news'] = data['news'].map(lambda x: clean(x))
	data.to_pickle('./data/dataframe.pickle')

encoder = LabelBinarizer()
encoder.fit(data["category"])
Y = encoder.transform(data["category"])

if os.path.isfile('./data/tokenizer.pickle'):
    tokenizer = Tokenizer()
    with open('./data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(data["news"])
    with open('./data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X = tokenizer.texts_to_matrix(data["news"], mode='tfidf')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Model ----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=x_train.shape[1:]))
model.add(Dense(y_train.shape[1], activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
