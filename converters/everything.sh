#!/bin/bash
# Convert a keras model to EVERYTHING

# Keras to TensorflowJS
./keras_to_tfjs.sh $1 $2
# Keras to Tensorflow
./keras_to_tf.sh $1 $2
# Zip it
./zip_results.zh $1
