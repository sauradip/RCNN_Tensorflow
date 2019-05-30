import tensorflow as tf
import os
import sys
from tensorflow.python.platform import gfile
# from tensorflow.python.platform import gfile
import numpy as np
from scipy.misc import imread, imresize

def classifer(imgs):
    with tf.Session() as persisted_sess:
        with gfile.FastGFile("/media/buiduchanh/Work/Project/Research/R2CNN/Classification/model/newmodel.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def)


        softmax_tensor = persisted_sess.graph.get_tensor_by_name('import/dense_1/Softmax:0')
        predictions = persisted_sess.run(softmax_tensor, {'import/input_1:0': imgs})
        predictions = np.squeeze(predictions)

        # top_k = predictions.argsort()[:][::-1]
        result = np.argsort(predictions)

        return np.argmax(predictions, axis=1)
        # print(top_k)
    