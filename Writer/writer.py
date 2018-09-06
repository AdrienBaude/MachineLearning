# Reference https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import keras.utils as ku
import numpy as np
import os, re

np.set_printoptions(threshold=np.nan)


def clean_data(data):
    data = data.lower()
    data = data.replace("’", "'")
    data = data.replace("'", "")
    data = data.replace("\ufeff", "")
    data = data.replace("“", '"')
    data = data.replace("”", '"')
    data = data.replace(":", "")
    data = data.replace(";", "")
    data = data.replace("…", "")
    data = data.replace("\n", " ")
    data = re.sub('(?<! )(?=[.,!?"])|(?<=[.,!?"])(?! )', r' ', data)
    return data


def dataset_preparation(data):
    corpus = data.lower().split(" ")
    corpus = list(filter(lambda x: x != '', corpus))
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    X = []
    Y = []
    for i in range(0, len(corpus) - seq_length, 1):
        token_list = tokenizer.texts_to_sequences([corpus[i:i + seq_length + 1]])[0]
        X.append(token_list[:seq_length - 1])
        Y.append(token_list[seq_length - 1])
    Y = ku.to_categorical(Y, num_classes=total_words)

    return np.asarray(X), np.asarray(Y), total_words


def create_model(total_words):
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=seq_length - 1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def generate_text(seed_text, next_words, model):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = np.asarray([token_list])
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text = seed_text.split(" ")
        seed_text = seed_text[1:]
        seed_text.append(output_word)
        seed_text = " ".join(seed_text)
        result += " " + output_word

    return result


data = (open(os.getcwd() + "/story.txt", encoding='utf-8').read())
data = clean_data(data)

tokenizer = Tokenizer(filters="")
seq_length = 10

X, Y, total_words = dataset_preparation(data)

if os.path.isfile('writer.h5'):
    model = create_model(total_words)
    model.load_weights("writer.h5")
else:
    model = create_model(total_words)
    model.fit(X, Y, epochs=100, verbose=1)
    model.save_weights("writer.h5")

data = "As Sombra fell silent, Twilight did as he suggested"
data = clean_data(data)
text = generate_text(data, 400, model)
print(text)
