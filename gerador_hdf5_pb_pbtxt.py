import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import logging

import keras
import keras.backend as K

hdf5_path = './pretrained/pre_trained_model.hdf5'
pb_and_pbtxt_destiny = './pretrained'


def dice_coef(y_true, y_pred, smooth=1000.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


model = keras.models.load_model(hdf5_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

model.summary()
K.set_learning_phase(0)
sess = K.get_session()

orig_output_node_names = [node.op.name for node in model.outputs]
print(orig_output_node_names)
converted_output_node_names = orig_output_node_names
constant_graph = graph_util.convert_variables_to_constants( sess, \
                            sess.graph.as_graph_def(), converted_output_node_names)
graph_io.write_graph(constant_graph, pb_and_pbtxt_destiny, "pre_trained_model.pb", as_text=False)
tf.train.write_graph(constant_graph, pb_and_pbtxt_destiny,"pre_trained_model.pbtxt", as_text=True)
logging.info('Saved the graph definition in ascii format at %s', "./pre_trained_model.pbtxt")
logging.info('Saved the freezed graph at %s', "./pre_trained_model.pb")

