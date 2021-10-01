# TESKD

By Zheng Li<sup>[1,4]</sup>, Xiang Li<sup>[2]</sup>, Lingfeng Yang<sup>[2,4]</sup>, Jian Yang<sup>[2]</sup>, Zhigeng Pan<sup>[3]</sup>*.

<sup>[1]</sup>Hangzhou Normal University, <sup>[2]</sup>Nanjing University of Science and Technology, <sup>[3]</sup>Nanjing University of Information Science and Technology, <sup>[4]</sup>MEGVII Technology

Email: lizheng1@stu.hznu.edu.cn

## Implementation

This is the official pytorch implementation for [Student Helping Teacher: Teacher Evolution via Self-Knowledge Distillation]()(TESKD)

In ths repository, all the models are implemented by Pytorch.

## Requirements
- Python3
- Pytorch >=1.7.0
- torchvision >= 0.8.1
- numpy >=1.18.5
- tqdm >=4.47.0

## Training 

In this code, you can reproduce the experiment results of classification task in the paper, including CIFAR-100 ang ImageNet.
Example training settings are for ResNet18 on CIFAR-100.

- Running TESKD.
~~~
python classification/main.py \
      --data_dir 'your_data_directory'\
      --final_dir 'your_storage_directory'\
      --name 'res18_our_cifar'\
      --model_name 'resnet_our'\
      --network_name 'cifarresnet18'\
      --data 'CIFAR100' \
      --batch_size 128 \
      --ce_weight 0.2 \
      --kd_weight 0.8 \
      --fea_weight 1e-7
~~~
