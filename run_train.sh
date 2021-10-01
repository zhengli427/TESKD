#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
      
python classification/main.py \
      --data_dir '/your_data_directory'\
      --final_dir '/your_storage_directory'\
      --name 'res18_our_cifar'\
      --model_name 'resnet_our'\
      --network_name 'cifarresnet18'\
      --data 'CIFAR100' \
      --batch_size 128 \
      --ce_weight 0.2 \
      --kd_weight 0.8 \
      --fea_weight 1e-7


