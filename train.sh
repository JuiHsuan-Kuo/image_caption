#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='3' python main.py \
    --data_dir ./data/data \
    --model_dir ./data/exp2 \
    --vocab ./data/data/vocab.pkl \
    --mode train \
    --num_epochs 50 \
    --epochs_per_eval 1 \
    --batch_size 128 
