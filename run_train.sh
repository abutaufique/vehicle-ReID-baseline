#!/bin/bash
python train_xent_tri.py \
    -s veri \
    -t veri \
    --root /localscratch \
    --height 256 \
    --width 256 \
    --optim amsgrad \
    --lr 0.0003 \
    --max-epoch 60 \
    --stepsize 20 40 \
    --train-batch-size 64 \
    --test-batch-size 100 \
    -a resnet50 \
    --save-dir log/resnet50-veri_partloss \
    --gpu-devices 0 \
