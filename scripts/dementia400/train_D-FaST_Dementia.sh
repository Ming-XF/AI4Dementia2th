#!/bin/bash


python main.py --wandb_entity cwg --project Dementia --model "DFaST" --num_repeat 3 --dataset 'Dementia400' --data_dir "../data/Dementia400/Dementia400.npy" --sparsity 0.6 --batch_size 16 --num_workers 5 --num_epochs 200 --frequency 128 --num_kernels 64 --window_size 32 --D 64 --p1 4 --p2 8 --drop_last False --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test
