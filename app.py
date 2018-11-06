from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import keras
import os
import json
from keras.models import load_model

from keras.preprocessing import image

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


img_path = './models/test_img.png'
img = image.load_img(img_path, target_size=(224, 224))
xx = image.img_to_array(img)

# y = xx.tolist()
# my_json_string = json.dumps(y)
#
# tmp = np.reshape(y,(-1))
# my_json_string = json.dumps(tmp.tolist())


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
    data = request.get_json()['features']
    x = json.loads(data)
    # x = json.loads(my_json_string)
    # text = json.loads(data)

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
    # return pred_label_test + ' ' + str(text)

if __name__ == '__main__':
    # print(os.listdir())
    app.run(debug=True)
