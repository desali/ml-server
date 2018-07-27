# coding: utf-8
import pandas as pd
from flask import Flask, jsonify, redirect, request
import tflearn
import tensorflow as tf
import requests
import json
import numpy as np
import re

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import  RussianStemmer
from nltk.tokenize import TweetTokenizer

VOCAB_SIZE = 5000
negative_tweets = pd.read_json("negative_hack.json", encoding = 'utf-8')
positive_tweets = pd.read_json("positive_hack1.json", encoding = 'utf-8')

stemer = RussianStemmer()
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}

def get_stem(token):
    stem = stem_cache.get(token, None)
    if stem:
        return stem
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem

stem_count = Counter()
tokenizer = TweetTokenizer()
num_texts1 = positive_tweets["text"].size
num_texts2 = negative_tweets["text"].size

def count_unique_tokens_in_tweets_pos(tweets):
    for i in range(0,num_texts1):
        tweet = positive_tweets["text"][i]
        tokens = tokenizer.tokenize(tweet)
        for token in tokens:
            stem = get_stem(token)
            stem_count[stem] += 1

def count_unique_tokens_in_tweets_neg(tweets):
    for i in range(0,num_texts2):
        tweet = negative_tweets["text"][i]
        tokens = tokenizer.tokenize(tweet)
        for token in tokens:
            stem = get_stem(token)
            stem_count[stem] += 1

count_unique_tokens_in_tweets_neg(neg)
count_unique_tokens_in_tweets_pos(pos)
vocab = sorted(stem_count, key=stem_count.get, reverse=True)[:VOCAB_SIZE]

def tweet_to_vector(tweet, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(tweet):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
    return vector

    tweet_vectors = np.zeros(
    (len(neg) + len(pos), VOCAB_SIZE), 
    dtype=np.int_)
tweets = []
for ii, (_, tweet) in enumerate(negative_tweets.iterrows()):
    tweets.append(tweet[0])
    tweet_vectors[ii] = tweet_to_vector(tweet[0])
for ii, (_, tweet) in enumerate(positive_tweets.iterrows()):
    tweets.append(tweet[0])
    tweet_vectors[ii + len(neg)] = tweet_to_vector(tweet[0])

labels = np.append(
    np.zeros(len(neg), dtype=np.int_), 
    np.ones(len(pos), dtype=np.int_))

labels[:10]
labels[-10:]

X = tweet_vectors
y = to_categorical(labels, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

app = Flask(__name__)

@app.route('/get_data', methods=['POST'])
def predict_data():
    if request.method == 'POST':
        model = build_model(learning_rate=0.75)
        model = model.load("model.tfl")

        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        source = request.args.get('source')
        keyword = request.args.get('keyword')

        req = requests.post("http://localhost:3000/api/v1/get_data", data={'start_date': '01.01.2018', 'end_date': '10.01.2018', 'source': 'Vk', 'keyword': 'Астана'})
        response = req.json()
        print(response)
        # response = request.values

        # {
        #   'keyword': params[:keyword],
        #   'date': params[:start_date],
        #   'sources': [
        #     {
        #       'vk': {
        #         'posts_count': @posts_needed.length,
        #         'posts_count_pos': @posts_needed.length,
        #         'posts_count_neg': @posts_needed.length,
        #       }
        #     }
        #   ]
        # }
        return start_date

#Building the NN
def build_model(learning_rate=0.1):
    tf.reset_default_graph()

    net = tflearn.input_data([None, 5000])
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
    app.run()
