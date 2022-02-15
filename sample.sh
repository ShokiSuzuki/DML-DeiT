#!/bin/bash


python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
        --models deit_tiny_patch16_224 deit_tiny_patch16_224 \
        --smoothing 0.0 --drop-path 0.0 \
        --no-repeated-aug --mixup 0.0 --cutmix 0.0 --reprob 0.0 --aa None \
        --batch-size 256 --data-path ~/dataset/imagenet --output_dir exp_`date "+%Y%m%d%H%M"`
