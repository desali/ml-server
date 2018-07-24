# coding: utf-8

from flask import Flask, jsonify, redirect, request
import tflearn
import tensorflow as tf
import requests
import json
import numpy as np

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
