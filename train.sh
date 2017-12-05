#! /usr/bin/env bash
CUDA_DEVICES_VISIBLE='2' python main.py \
    --data_dir /media/VSlab3/fionakuo/CV_FINAL/data \
    --model_dir /media/VSlab3/fionakuo/CV_FINAL/exp \
    --vocab /media/VSlab3/fionakuo/CV_FINAL/data/vocab.pkl \
    --mode train \
    --num_epochs 500 \
    --batch_size 128 
