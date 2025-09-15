#!/bin/bash

python main.py --model "STAGIN" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --percentage 1. --batch_size 16 --num_epochs 200 --drop_last False --d_model 64 --window_size 50 --window_stride 3 --dynamic_length 440 --num_heads 1 --num_layers 2 --learning_rate 0.0005 --max_learning_rate 0.001 --schedule 'one_cycle' --do_train --do_evaluate --do_test




