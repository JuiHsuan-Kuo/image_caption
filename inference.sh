#! /usr/bin/env bash
CUDA_VISIBLE_DEVICES='3' python main.py \
    --data_dir ./data/data \
    --model_dir ./data/exp \
    --mode inference \
    --vocab ./data/data/vocab.pkl \
    --predict_image test.jpg
