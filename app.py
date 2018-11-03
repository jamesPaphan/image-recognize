from flask import Flask
from flask import request
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/post', methods=['POST'])
def index():
    #grabs the data tagged as 'name'
    # name = request.form['name']
    name = request.get_json()['name']

    #sending a hello back to the requester
    return "Hello " + name

@app.route('/predict', methods=['POST'])
def predict():
    frozen_graph_filename = 'tensorflow_inception_graph.pd'
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.get_default_graph()
    tf.import_graph_def(graph_def, name='')

    batch = graph.get_tensor_by_name('input:0')
    prediction = graph.get_tensor_by_name('output:0')

    x = request.get_json()['features']
    return x

    # with tf.Session() as sess:
    #     img_path = 'test_img.png'
    #     img = image.load_img(img_path, target_size=(224, 224))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x)
    #
    #     values = sess.run(prediction, feed_dict={batch: x})
    #
    #     pred_class_test = np.argmax(values)
    #     pred_label_test = idx2label[pred_class_test]
    #     print('Prediction :{}, confidence : {:.3f}'.format(
    #         pred_label_test,
    #         values[0][pred_class_test]))


if __name__ == '__main__':
    app.run()
