from tensorflowjs import converters
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from zipfile import ZipFile
import os


keras_model = './path/to/load/the/keras/model/file.h5' # path of the Keras model file
path_of_tfjs_dir = '.path/to/save/the/tensorflowjs_file' # path of the Tensorflow JS directory
path_of_tflite_dir = './path/to/tensorflowjs_file' # path of the Tensorflow LITE directory
path_of_tf_dir = '.path/to/save/the/tensorflow_file' # path of the Tensorflow directory

zip_dir = './path/to.zip/all/the/models' #path of the zip directiory

tf.compat.v1.disable_eager_execution()

model = load_model(keras_model)

# Convert Keras to Tensorflow
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def keras2tf(path_of_keras_file, path_of_tf_dir, file='\file_name.pb'):
    model = load_model(path_of_keras_file)
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf_pb = tf.train.write_graph(frozen_graph, path, file, as_text=False)
    return (tf_pb)


def keras2tflite(path_of_keras_file, path_of_tflite_dir, tflite_file_name = '\file_name.tflite'):
    # Convert Keras to TFLITE
    converter = tf.lite.TFLiteConverter.from_keras_model_file(path_of_keras_file)
    tfmodel = converter.convert()
    tflite = open(path_of_tflite_dir + tflite_file_name,"wb").write(tfmodel)
    return tflite

def keras2tfjs(path_of_keras_file, path_of_tfjs_dir):
    # Convert Keras to Tensorflow JS
    # If tensorflow JS library missing the dispatch_keras_h5_to_tfjs_layers_model_conversion function in init.py add the function
    tf_js = converters.dispatch_keras_h5_to_tfjs_layers_model_conversion(path_of_keras_file, path_of_tfjs_dir)
    return tf_js

# Zip all the Files
zipObj = ZipFile(zip_dir + r'file.zip', 'w')

# Add multiple files to the zip
zipObj.write(path_of_tflite_dir + r'\file.tflite')
zipObj.write(path_of_tf_dir + r'\file.pb')
zipObj.write(path_of_tfjs_dir + r'\file.json')

# Close the Zip File
zipObj.close()

# Removing the files after Zip
os.remove(path_of_tflite_dir + r'\file.tflite')
os.remove(path_of_tf_dir + r'\file.pb')
os.remove(path_of_tfjs_dir + r'\file.json')
