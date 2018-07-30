# coding: utf-8

from flask import Flask, jsonify, redirect, request
from flask_cors import CORS

import tflearn
import tensorflow as tf
import requests
import json
import numpy as np
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import  RussianStemmer
from nltk.tokenize import TweetTokenizer

app = Flask(__name__)
CORS(app)

@app.route('/get_data', methods=['POST'])
def predict_data():
    global vectorized_parsed_tweets
    global parsed_tweets
    global model

    init()

    if request.method == 'POST':
        data = request.get_json()
        start_date = data["startdate"]
        end_date = data["enddate"]
        keyword = data["keyword"]

        req = requests.post("https://social-back.herokuapp.com/api/v1/get_data", data={'start_date': start_date, 'end_date': end_date, 'keyword': keyword})
        response = req.json()

        model = build_model(learning_rate=0.75)
        model.load("./my_model.tflearn")

        response_object = {
            'line': {
                'post': {
                    'labels': [],
                    'datasets': [{
                        'label': 'Инстаграм',
                        'data': [],
                        'pointBackgroundColor': 'blue',
                        'borderWidth': 3,
                        'borderColor': 'blue',
                        'pointBorderColor': 'blue',
                        'fill': False
                    }]
                },
                'comment': {
                    'labels': [],
                    'datasets': [{
                        'label': 'Инстаграм',
                        'data': [],
                        'pointBackgroundColor': 'blue',
                        'borderWidth': 3,
                        'borderColor': 'blue',
                        'pointBorderColor': 'blue',
                        'fill': False
                    }]
                }
            },
            'pie': {
                'post': {
                    'insta': {
                    'labels': ['Позитивность', 'Негативность'],
                     'datasets': [{
                         'backgroundColor': [
                           '#00D8FF',
                           '#E46651'
                         ],
                         'data': []
                       }]
                    }
               },
               'comment': {
                  'insta': {
                   'labels': ['Позитивность', 'Негативность'],
                    'datasets': [{
                        'backgroundColor': [
                          '#00D8FF',
                          '#E46651'
                        ],
                        'data': []
                    }]
                 }
              }
            }
            # 'main': [{
            #     'user': '',
            #     'posts': []
            #     }
            # ]
        }

        pos = 0
        neg = 0

        posts = response['posts']

        for day in posts:
            array = []

            for res in posts[day]:
                array.append(np.fromstring(res["vector"], dtype=int, sep=' '))

            predictions = (np.array(model.predict(array))[:,0] >= 0.5).astype(np.int_)

            # pos = 0
            # neg = 0
            for prediction in predictions:
                if prediction == 1:
                    pos += 1
                else:
                    neg += 1


            response_object['line']['post']['labels'].append(day)
            response_object['line']['post']['datasets'][0]['data'].append(len(predictions))


        response_object['line']['post']['datasets'][0]['label'] = keyword

        # response_object['pie']['post']['insta']['datasets'][0]['data'].append( (pos / (pos + neg)) * 100 )
        response_object['pie']['post']['insta']['datasets'][0]['data'].append( pos )
        # response_object['pie']['post']['insta']['datasets'][0]['data'].append( (neg / (pos + neg)) * 100 )
        response_object['pie']['post']['insta']['datasets'][0]['data'].append( neg )

        pos = 0
        neg = 0

        comments = response['comments']

        for day in comments:
            array = []

            for res in comments[day]:
                array.append(np.fromstring(res["vector"], dtype=int, sep=' '))

            predictions = (np.array(model.predict(array))[:,0] >= 0.5).astype(np.int_)

            # pos = 0
            # neg = 0
            for prediction in predictions:
                if prediction == 1:
                    pos += 1
                else:
                    neg += 1


            response_object['line']['comment']['labels'].append(day)
            response_object['line']['comment']['datasets'][0]['data'].append(len(predictions))


        response_object['line']['comment']['datasets'][0]['label'] = keyword

        # response_object['pie']['post']['insta']['datasets'][0]['data'].append( (pos / (pos + neg)) * 100 )
        response_object['pie']['comment']['insta']['datasets'][0]['data'].append( pos )
        # response_object['pie']['post']['insta']['datasets'][0]['data'].append( (neg / (pos + neg)) * 100 )
        response_object['pie']['comment']['insta']['datasets'][0]['data'].append( neg )

        return jsonify(response_object)

@app.route('/vectorize', methods=['POST'])
def vectorize():
    init()

    if request.method == 'POST':
        print("Request Request Request Request Request")
        vectorized_parsed_tweets = ""

        parsed_tweets = request.json

        for i in range(0, len(parsed_tweets)):
            if( i == len(parsed_tweets) - 1 ):
                for i in tweet_to_vector(parsed_tweets[i]["text"].lower(), True):
                    vectorized_parsed_tweets += str(i) + ' '
            else:
                for i in tweet_to_vector(parsed_tweets[i]["text"].lower(), True):
                    vectorized_parsed_tweets += str(i) + ' '
                vectorized_parsed_tweets += ','

        print(vectorized_parsed_tweets)

        return vectorized_parsed_tweets


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
        # elif show_unknowns:
            # print("Unknown token: {}".format(token))
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
