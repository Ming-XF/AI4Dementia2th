#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3

cd ../..
python main.py \
\
--project "FaST-P-C42B1" \
--model "DFaST" \
--num_repeat 5 \
\
--dataset "C42B" \
--data_dir "../data/C42B/C42B128.npy" \
--sparsity 1 \
--batch_size 32 \
--num_epochs 200 \
--frequency 128 \
--num_kernels 64 \
--window_size 3 \
--D 1 \
--p1 8 \
--p2 16 \
--drop_last True \
--num_heads 4 \
--distill \
--mix_up \
--num_layers 1 \
--learning_rate 1e-3 \
--dropout 0.5 \
--schedule 'cos' \
\
--do_train \
--do_evaluate \
--do_test
