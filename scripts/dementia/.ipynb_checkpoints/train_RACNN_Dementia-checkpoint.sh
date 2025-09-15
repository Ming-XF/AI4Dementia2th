#!/bin/bash

python main.py --wandb_entity cwg --project Dementia --model "RACNN" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --batch_size 16 --k 2 --num_epochs 200 --drop_last False --schedule 'cos' --learning_rate 1e-4 --do_train --do_evaluate --do_test



