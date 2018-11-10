from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import keras
import os
import json
import base64
from keras.models import load_model

app = Flask(__name__)

#______________________ INITIAL _________________________#

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph

batch = {}
prediction = {}
labels = {}

for file in os.listdir("./models"):
    if file.endswith(".pb"):
        frozen_graph_filename = './models/' + file
        _class = file[:-3]
        graph = load_graph(frozen_graph_filename)
        batch[_class] = graph.get_tensor_by_name('input:0')
        prediction[_class] = graph.get_tensor_by_name('output:0')

        with open('./models/'+_class+'.txt', 'r') as f:
            label = f.read()
            labels[_class] = label.split('\n')

###########################################################

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/post', methods=['POST'])
def index():
    #grabs the data tagged as 'name'
    # name = request.form['name']
    # features = request.form['features']
    # name = request.get_json()['name']

    #sending a hello back to the requester
    return features

@app.route('/predict', methods=['POST'])
def predict():
    model = request.get_json()['model']

    features_string_base64 = request.get_json()['features']                     #got string of base64 : 'YWJj'
    features_byte = base64.b64decode(bytes(features_string_base64, "utf-8"))    #'YWJj' -> b'YWJj' -> b'abc'

    features = np.frombuffer(features_byte, dtype=np.uint8 ,count=224*224*3)    #convert byte to array of int

    with tf.Session(graph=graph) as sess:
        features = np.reshape(features, (224,224,3))
        features = np.expand_dims(features, axis=0)
        values = sess.run(prediction[model], feed_dict={batch[model]: features})

        pred_class_test = np.argmax(values)
        pred_label_test = labels[model][pred_class_test]

    return json.dumps({'label': pred_label_test, 'confidence': str(values[0][pred_class_test])})

if __name__ == '__main__':
    # print(os.listdir())
    app.run(debug=True)
