from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import keras
import os
import json
import base64
from keras.models import load_model

app = Flask(__name__)

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph

frozen_graph_filename = './models/tensorflow_inception_graph.pb'
graph = load_graph(frozen_graph_filename)
batch = graph.get_tensor_by_name('input:0')
prediction = graph.get_tensor_by_name('output:0')

f = open('./models/imagenet_comp_graph_label_strings.txt', 'r')
labels = f.read()
labels = labels.split('\n')
f.close()

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/post', methods=['POST'])
def index():
    #grabs the data tagged as 'name'
    name = request.form['name']
    features = request.form['features']
    # name = request.get_json()['name']

    #sending a hello back to the requester
    return features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()['features']
    features_base64 = json.loads(data)
    # name = request.get_json()['name']

    # features_base64 = request.form['features']
    features_byte = base64.b64decode(features_base64.encode("utf-8"))
    # return features_byte.decode("utf-8")[:10]
    features_string = features_byte.decode("utf-8")
    return str(len(features_string))
    x = np.zeros(224*224*3)

    for i in range(0, 224*224*3):
        x[i] = ord(features_string[i])

    with tf.Session(graph=graph) as sess:
        x = np.reshape(x,(224,224,3))
        x = np.expand_dims(x, axis=0)
        values = sess.run(prediction, feed_dict={batch: x})

        pred_class_test = np.argmax(values)
        pred_label_test = labels[pred_class_test]
        # print('Prediction :{}, confidence : {:.3f}'.format(
        #     pred_label_test,
        #     values[0][pred_class_test]))
    return json.dumps({'label': pred_label_test, 'confidence': str(np.max(values))})
    # return pred_label_test+' '+name

if __name__ == '__main__':
    # print(os.listdir())
    app.run(debug=True)
