# coding: utf-8

from flask import Flask, jsonify, redirect, request
import tflearn
import tensorflow as tf
import requests
import json
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import  RussianStemmer
from nltk.tokenize import TweetTokenizer

app = Flask(__name__)

@app.route('/get_data', methods=['POST'])
def predict_data():
    global vectorized_parsed_tweets
    global parsed_tweets
    global model

    init()

    if request.method == 'POST':
        model = build_model(learning_rate=0.75)
        model.load("./my_model.tflearn")
        predictions = (np.array(model.predict(vectorized_parsed_tweets))[:,0] >= 0.5).astype(np.int_)

        # for i in range(0, len(vectorized_parsed_tweets)):
        #     print(parsed_tweets["text"][i])
        #     print(predictions[i])

        return predictions

@app.route('/vectorize', methods=['POST'])
def vectorize():
    global parsed_tweets
    global vectorized_parsed_tweets

    init()

    if request.method == 'POST':
        print("Request Request Request Request Request")

        parsed_tweets = pd.read_json("test_posts.json", encoding = 'utf-8')
        vectorized_parsed_tweets = []

        for i in range(0, parsed_tweets["text"].size):
            vectorized_parsed_tweets.append(tweet_to_vector(parsed_tweets["text"][i].lower(), True))

        print(vectorized_parsed_tweets)

        return str(vectorized_parsed_tweets)


def init():
    global VOCAB_SIZE
    global token_2_idx
    global stem_cache
    global stemer
    global regex

    VOCAB_SIZE = 5000
    stem_vocab = pd.read_json("stem_vocab.json", encoding = 'utf-8')
    token_2_idx = {stem_vocab.values[i][0]: i for i in range(VOCAB_SIZE)}

    with open('stem_cache.json') as f:
        stem_cache = json.load(f)

    stemer = RussianStemmer()
    regex = re.compile('[^а-яА-Я ]')

def get_stem(token):
    global stem_cache
    global stemer
    global regex

    token = regex.sub('', token).lower()
    token = stemer.stem(token)
    stem = stem_cache.get(token, None)

    if stem:
        return stem

def tweet_to_vector(tweet, show_unknowns=False):
    global VOCAB_SIZE
    global token_2_idx

    tokenizer = TweetTokenizer()

    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(tweet):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif show_unknowns:
            print("Unknown token: {}".format(token))
    return vector

#Building the NN
def build_model(learning_rate=0.1):
    global VOCAB_SIZE

    tf.reset_default_graph()

    net = tflearn.input_data([None, VOCAB_SIZE])
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net,
        optimizer='sgd',
        learning_rate=learning_rate,
        loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model

if __name__ == "__main__":
    app.run(debug=True)
