#!/bin/bash
# Convert a keras model to TFJS

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
RESULT_DIR="results/TFJS_${NAME}"

# Regular
tensorflowjs_converter --input_format keras $1/$2 $1/$RESULT_DIR/tfjs_$NAME
# Quantized
tensorflowjs_converter --quantize_uint8 --input_format keras $1/$2 $1/$RESULT_DIR/tfjs_quant_uint8_$NAME
tensorflowjs_converter --quantize_uint16 --input_format keras $1/$2 $1/$RESULT_DIR/tfjs_quant_uint16_$NAME
tensorflowjs_converter --quantize_float16 --input_format keras $1/$2 $1/$RESULT_DIR/tfjs_quant_float16_$NAME


