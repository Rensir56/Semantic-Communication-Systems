#!/bin/bash

RATE=1.0
MODEL_ENCODER="MLP_MNIST_encoder_combining_1.000000.pkl"
MODEL_CLASSIFIER="MLP_MNIST.pkl"
DATASET_PATH="./dataset/mnist"
OUTPUT_IMAGE_PATH="./image_recover_combing"

echo "Compression Rate: $RATE"
echo "Encoder Model Path: $MODEL_ENCODER"
echo "Classifier Model Path: $MODEL_CLASSIFIER"
echo "Dataset Path: $DATASET_PATH"
echo "Output Image Path: $OUTPUT_IMAGE_PATH"

python box.py \
    --rate $RATE \
    --model_encoder $MODEL_ENCODER \
    --model_classifier $MODEL_CLASSIFIER \
    --dataset_path $DATASET_PATH \
    --output_image_path $OUTPUT_IMAGE_PATH
