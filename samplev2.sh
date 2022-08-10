#!/bin/bash


python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 --use_env train.py \
    --models deit_small_patch16_LS --data-path ~/dataset/imagenet --batch 64 --lr 4e-3 --epochs 400 --weight-decay 0.05 --sched cosine --input-size 224 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt fusedlamb --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment --output_dir exp/test
