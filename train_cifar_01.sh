#!/bin/bash

python3 train_cifar.py --lr 1e-4 --epoch 3 --bs 1000 --mnbs 50 --C 1.0 --algo DPSGD --tag B1 --model vit_small_patch16_224