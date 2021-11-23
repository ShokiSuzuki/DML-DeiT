# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from model.loss import DMLLoss
import utils


def train_one_epoch(models, criterion: DMLLoss, data_loader: Iterable, optimizers,
                    device: torch.device, epoch: int, models_ema,
                    loss_scalers, clip_grad=None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):

    metric_loggers = []
    for model in models:
        model.train(set_training_mode)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_loggers.append(metric_logger)

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_loggers[0].log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = [model(samples) for model in models]

        ## model training
        for i, (model, optimizer, loss_scaler, model_ema) in enumerate(zip(models,
                                                                            optimizers,
                                                                            loss_scalers,
                                                                            models_ema)):
            with torch.cuda.amp.autocast():
                loss = criterion(i, outputs, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            # loss_scaler.scale(loss).backward()
            # if clip_grad is not None:
            #     loss_scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(models[i].parameters(), clip_grad=clip_grad)
            # loss_scaler.step(optimizer)
            # loss_scaler.update()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=clip_grad,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    ave = []
    # gather the stats from all processes
    for i in range(len(metric_loggers)):
        metric_loggers[i].synchronize_between_processes()
        print("Averaged stats:", metric_loggers[i])

        ave.append({k: meter.global_avg for k, meter in metric_loggers[i].meters.items()})

    return ave


@torch.no_grad()
def evaluate(data_loader, models, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_loggers = []
    for model in models:
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_loggers.append(metric_logger)

    header = 'Test:'

    for images, target in metric_loggers[0].log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_size = images.shape[0]
        for i, model in enumerate(models):
            with torch.cuda.amp.autocast():
                output = model(images)
                loss   = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            metric_loggers[i].update(loss=loss.item())
            metric_loggers[i].meters['acc1'].update(acc1.item(), n=batch_size)
            metric_loggers[i].meters['acc5'].update(acc5.item(), n=batch_size)

    ave = []
    # gather the stats from all processes
    for i in range(len(metric_loggers)):
        metric_loggers[i].synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_loggers[i].acc1, top5=metric_loggers[i].acc5, losses=metric_loggers[i].loss))

        ave.append({k: meter.global_avg for k, meter in metric_loggers[i].meters.items()})

    return ave
