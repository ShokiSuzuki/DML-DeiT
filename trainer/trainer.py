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
                    max_norm: float = 0, mixup_fn: Optional[Mixup] = None,
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

        ## model training (model0 is independent)
        for i in range(len(models)):
            with torch.cuda.amp.autocast():
                loss = criterion(i, outputs, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizers[i].zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizers[i], 'is_second_order') and optimizers[i].is_second_order
            # loss_scalers[i](loss, optimizers[i], clip_grad=max_norm,
            #                 parameters=models[i].parameters(), create_graph=is_second_order)
            loss.backward(create_graph=is_second_order)
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm=max_norm)
            optimizers[i].step()

            if models_ema[i] is not None:
                models_ema[i].update(models[i])

            metric_loggers[i].update(loss=loss_value)
            metric_loggers[i].update(lr=optimizers[i].param_groups[0]["lr"])

    ave = []
    # gather the stats from all processes
    for metric_logger in metric_loggers:
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        ave.append({k: meter.global_avg for k, meter in metric_logger.meters.items()})

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

    for images, target in metric_logger.log_every(data_loader, 10, header):
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
    for metric_logger in metric_loggers:
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        ave.append({k: meter.global_avg for k, meter in metric_logger.meters.items()})

    return ave
