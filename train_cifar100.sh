#!/bin/bash

python3 train_cifar100.py --lr 0.1 --epoch 2 --bs 1000 --mnbs 50 --C 1.0 0.1 0.01 --C2 1.0 2.0 5.0 10.0 30.0 --algo DiceSGD --tag Abl_C1C2 --model vit_small_patch16_224