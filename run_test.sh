#!/bin/bash
python train_xent_tri.py \
    -s veri \
    -t veri \
    --root /home/at7133/dataset/EgTest_crops \
    --height 128 \
    --width 256 \
    --test-batch-size 100 \
    --evaluate \
    -a resnet50 \
    --load-weights log/resnet50-veri/model.pth.tar-60 \
    --save-dir log/eval-veri-to-vehicleID \
    --gpu-devices 0 \
    --test-size 800 \
