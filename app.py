from flask import Flask, request
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/post', methods=['POST'])
def index():
    #grabs the data tagged as 'name'
    # name = request.form['name']
    name = request.get_json()['name']

    #sending a hello back to the requester
    return "Hello " + name

@app.route('/predict', methods=['POST'])
def predict():
    x = request.get_json()['features']

    with tf.Session(graph=graph) as sess:
        x = np.expand_dims(x, axis=0)
        values = sess.run(prediction, feed_dict={batch: x})

        pred_class_test = np.argmax(values)
        # pred_label_test = idx2label[pred_class_test]
        # print('Prediction :{}, confidence : {:.3f}'.format(
        #     pred_label_test,
        #     values[0][pred_class_test]))
    return pred_class_test

if __name__ == '__main__':
    # print(os.listdir())
    frozen_graph_filename = './models/tensorflow_inception_graph.pb'
    graph = load_graph(frozen_graph_filename)
    batch = graph.get_tensor_by_name('input:0')
    prediction = graph.get_tensor_by_name('output:0')

    app.run(debug=True)
