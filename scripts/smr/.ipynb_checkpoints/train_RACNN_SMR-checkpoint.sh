#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2

cd ../..
python main.py \
--wandb_entity cwg \
--project SMR \
\
--model "RACNN" \
--num_repeat 5 \
\
--dataset 'SMR' \
--data_dir "/data/datasets/SMR/SMR128.npy" \
--batch_size 32 \
--k 2 \
--num_epochs 200 \
--drop_last True \
--schedule 'cos' \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test


python main.py --wandb_entity cwg --project Dementia --model "RACNN" --num_repeat 3 --dataset 'Dementia' --data_dir "../data/Dementia200/Dementia200.npy" --batch_size 16 --k 2 --num_epochs 200 --drop_last False --schedule 'cos' --learning_rate 1e-4 --do_train --do_evaluate --do_test



