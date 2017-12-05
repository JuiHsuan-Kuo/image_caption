#! /usr/bin/env bash
python main.py \
    --data_dir /media/VSlab3/fionakuo/CV_FINAL/data \
    --model_dir /media/VSlab3/fionakuo/CV_FINAL/exp \
    --mode inference \
    --vocab /media/VSlab3/fionakuo/CV_FINAL/data/vocab.pkl \
    --predict_image test.jpg 
