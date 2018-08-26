# Reference https://www.opencodez.com/python/text-classification-using-keras.htm

import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
import os, nltk, string


def getDataFrame(files):
    label_index = files_train.target
    label_names = files_train.target_names
    labelled_files = files_train.filenames

    # stemmer = nltk.PorterStemmer()
    # table = str.maketrans('', '', string.punctuation)
    # texts = []
    # for f in labelled_files:
    #     texts.append(Path(f).read_text())
    #
    # texts = [text.replace('\n', ' ') for text in texts]
    # texts = [text.split(' ') for text in texts]
    # texts = [[word.translate(table) for word in text] for text in texts]
    # texts = [[word.lower() for word in text] for text in texts]
    # texts = [list(filter(lambda x: x.isalnum(), text)) for text in texts]
    # texts = [list(filter(lambda x: x not in nltk.corpus.stopwords.words('english'), text)) for text in texts]
    # texts = [list(map(stemmer.stem, text)) for text in texts]
    # texts = list(map(' '.join, texts))

    data_tags = ["filename", "category", "news"]
    data_list = []

    i = 0
    for f in labelled_files:
        # data_list.append((f, label_names[label_index[i]], texts[i]))
        data_list.append((f, label_names[label_index[i]], Path(f).read_text()))
        i += 1

    return pd.DataFrame.from_records(data_list, columns=data_tags)


files_train = skds.load_files(os.getcwd() + "\\20news-bydate-train", load_content=False)
files_test = skds.load_files(os.getcwd() + "\\20news-bydate-test", load_content=False)

data_train = getDataFrame(files_train)
data_test = getDataFrame(files_test)

encoder = LabelBinarizer()
encoder.fit(data_train["category"])
y_train = encoder.transform(data_train["category"])
y_test = encoder.transform(data_test["category"])

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(15000,)))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu", ))
model.add(Dropout(0.3))
model.add(Dense(20, activation="softmax", ))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if os.path.isfile('text.h5') and os.path.isfile('tokenizer.pickle'):
    tokenizer = Tokenizer()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    x_train = tokenizer.texts_to_matrix(data_train["news"], mode='tfidf')
    x_test = tokenizer.texts_to_matrix(data_test["news"], mode='tfidf')

    model.load_weights("text.h5")
else:
    tokenizer = Tokenizer(num_words=15000)
    tokenizer.fit_on_texts(data_train["news"])
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    x_train = tokenizer.texts_to_matrix(data_train["news"], mode='tfidf')
    x_test = tokenizer.texts_to_matrix(data_test["news"], mode='tfidf')

    model.fit(x_train, y_train, batch_size=100, epochs=30, verbose=1, validation_split=0.1)
    model.save_weights("text.h5")

score = model.evaluate(x_test, y_test, batch_size=100, verbose=1)
print('\nAccuracy: ', score[1])
