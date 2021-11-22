# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
from os import write
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma

from dataloader.datasets import build_dataset
from dataloader.samplers import RASampler
from model.loss import DMLLoss
from model.model import *
from trainer.trainer import *
import utils
from config import get_args_parser

from torch.utils.tensorboard import SummaryWriter


def main(args):
    utils.init_distributed_mode(args)

    if args.aa == 'None':
        args.aa = None

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = False


    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    models = []
    for model_name in args.models:
        print(f"Creating model: {model_name}")
        model = deit_tiny_distilled_patch16_224(
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path
        )

        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        models.append(model)


    models_ema = [None for _ in range(len(models))]
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        for i, model in enumerate(models):
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            models_ema[i] = model_ema


    ## settings for model
    models_without_ddp = [model for model in models]
    if args.distributed:
        for i, model in enumerate(models):
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            models_without_ddp[i] = model.module


    # display number of params in models
    n_parameters = []
    for model in models:
        n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameter)
        n_parameters.append(n_parameter)

    if args.data_set == 'IMNET':
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    optimizers    = []
    lr_schedulers = []
    for model_without_ddp in models_without_ddp:
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)

        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    output_dir = Path(args.output_dir)
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DMLLoss(
        criterion, args.distillation_type, args.distillation_alpha, args.distillation_tau, output_dir
    )


    if args.resume:
        for i in range(len(models)):
            checkpoint_path = args.resume + '/' + f'checkpoint{i}.pth'
            checkpoint = torch.load(checkpoint_path , map_location='cpu')
            models_without_ddp[i].load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizers[i].load_state_dict(checkpoint['optimizer'])
                lr_schedulers[i].load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if args.model_ema:
                    utils._load_checkpoint_for_ema(models_ema[i], checkpoint['model_ema'])

    if args.eval:
        test_stats_list = evaluate(data_loader_val, models, device)
        for test_stats in test_stats_list:
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    if args.use_tensorboard:
        writer = SummaryWriter()
        if utils.is_main_process():
            with (Path(writer.log_dir) / "args.txt").open("a") as f:
                print(args, file=f)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracys = [0.0] * len(models)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats_list = train_one_epoch(
            models, criterion, data_loader_train,
            optimizers, device, epoch,
            models_ema, args.clip_grad, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step(epoch)

        if args.output_dir:
            for i, (model_without_ddp, optimizer, lr_scheduler, model_ema) in enumerate(zip(models_without_ddp,
                                                                                optimizers,
                                                                                lr_schedulers,
                                                                                models_ema)):
                one_epoch_path = Path(args.output_dir + '/' + 'epoch' + f'{epoch}'.zfill(4))
                one_epoch_path.mkdir(parents=True, exist_ok=True)
                checkpoint_path = one_epoch_path / f'checkpoint{i}.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': None if model_ema is not None else get_state_dict(model_ema),
                    'args': args,
                }, checkpoint_path)


        test_stats_list = evaluate(data_loader_val, models, device)
        for i, test_stats in enumerate(test_stats_list):
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracys[i] = max(max_accuracys[i], test_stats["acc1"])
            print(f'Max accuracy: {max_accuracys[i]:.2f}%')

        for i, (train_stats, test_stats) in enumerate(zip(train_stats_list, test_stats_list)):
            if args.use_tensorboard:
                writer.add_scalars('acc/test', {'Ind': test_stats['acc1']}, epoch)
                writer.add_scalars('loss', {'train loss' : train_stats['loss'],  'test loss' : test_stats['loss']}, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters[i]}

            if args.output_dir and utils.is_main_process():
                with (output_dir / f"log{i}.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
