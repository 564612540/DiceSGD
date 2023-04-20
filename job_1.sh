#!/bin/bash

python3 ./main_fed_n.py --dataset emnist --percentage 0.9 --num_channels 1 --model alex --epochs 129 --gpu 0 --local_ep 25 --num_users 20 --frac 1 --local_bs 64 --lr 0.01 --dp --path 09

python3 ./main_fed_n.py --dataset emnist --percentage 0.9 --num_channels 1 --model alex --epochs 129 --gpu 0 --local_ep 25 --num_users 20 --frac 1 --local_bs 64 --lr 0.01 --dp --path 05

python3 ./main_fed_n.py --dataset emnist --percentage 0.9 --num_channels 1 --model alex --epochs 129 --gpu 0 --local_ep 25 --num_users 20 --frac 1 --local_bs 64 --lr 0.01 --dp --path 01

python3 ./main_fed_n.py --dataset emnist --percentage 0.9 --num_channels 1 --model resnet --epochs 129 --gpu 0 --local_ep 25 --num_users 20 --frac 1 --local_bs 64 --lr 0.01 --dp --path 09

python3 ./main_fed_n.py --dataset emnist --percentage 0.9 --num_channels 1 --model resnet --epochs 129 --gpu 0 --local_ep 25 --num_users 20 --frac 1 --local_bs 64 --lr 0.01 --dp --path 05

python3 ./main_fed_n.py --dataset emnist --percentage 0.9 --num_channels 1 --model resnet --epochs 129 --gpu 0 --local_ep 25 --num_users 20 --frac 1 --local_bs 64 --lr 0.01 --dp --path 01