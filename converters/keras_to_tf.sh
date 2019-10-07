#!/bin/bash
# Convert a keras model to Tensorflow

if [ "$1" == "" ]; then
    echo "First parameter is the path"
    exit 1
fi

if [ "$2" == "" ]; then
    echo "Second parameter is the file"
    exit 1
fi

# No name passed? Just use "model"
NAME=${3:-model}
RESULT_DIR="results/TF_${NAME}"
# Make Tensorflow Results Folder
mkdir $1/$RESULT_DIR
K2TF="/Users/gantman/Documents/Projects/ml/keras_to_tensorflow/keras_to_tensorflow.py"

# convert to Tensorflow (expects python3 and keras_to_tensorflow)
# https://github.com/amir-abdi/keras_to_tensorflow
python3 $K2TF --input_model=$1/$2 --output_model=$1/$RESULT_DIR/$NAME.pb
# Quantized version
python3 $K2TF --quantize=true --input_model=$1/$2 --output_model=$1/$RESULT_DIR/quant_$NAME.pb
