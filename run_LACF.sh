#!/bin/bash

# Make sure to run preprocess_airquality.py first if you haven't
# python3 preprocess_airquality.py

echo "Running LACFNet on Weather Dataset"
python3 run_longExp.py \
  --model LACFNet \
  --dataset Weather \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --itr 1

echo "Running LACFNet on AirQuality Dataset"
python3 run_longExp.py \
  --model LACFNet \
  --dataset AirQuality \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --itr 1
