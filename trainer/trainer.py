import math
import sys
from typing import Iterable, Optional, List

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from model.loss import DMLLoss
import utils


def train_one_epoch(models: List[torch.nn.Module], criterion: DMLLoss, data_loader: Iterable, optimizers: List[torch.optim.Optimizer],
                    device: torch.device, epoch: int, models_ema: List[ModelEma],
                    loss_scalers, clip_grad=None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):

    num_models = len(models)
    metric_loggers = []
    for i in range(num_models):
        models[i].train(set_training_mode)

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

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast():
            outputs = [models[i](samples) for i in range(num_models)]

        ## model training
        for i in range(num_models):
            with torch.cuda.amp.autocast():
                loss = criterion(i, outputs, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizers[i].zero_grad()
            is_second_order = hasattr(optimizers[i], 'is_second_order') and optimizers[i].is_second_order
            loss_scalers[i](loss, optimizers[i], clip_grad=clip_grad,
                            parameters=models[i].parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if models_ema[i] is not None:
                models_ema[i].update(models[i])

            metric_loggers[i].update(loss=loss_value)
            metric_loggers[i].update(lr=optimizers[i].param_groups[0]["lr"])

    ave = []
    # gather the stats from all processes
    for i in range(len(metric_loggers)):
        metric_loggers[i].synchronize_between_processes()
        print("Averaged stats:", metric_loggers[i])

        ave.append({k: meter.global_avg for k, meter in metric_loggers[i].meters.items()})

    return ave


@torch.no_grad()
def evaluate(data_loader: Iterable, models: List[torch.nn.Module], device: torch.device):
    criterion = torch.nn.CrossEntropyLoss()

    num_models = len(models)
    metric_loggers = []
    for i in range(num_models):
        models[i].eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_loggers.append(metric_logger)

    header = 'Test:'

    for images, target in metric_loggers[0].log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_size = images.shape[0]
        for i in range(num_models):
            with torch.cuda.amp.autocast():
                output = models[i](images)
                loss   = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            metric_loggers[i].update(loss=loss.item())
            metric_loggers[i].meters['acc1'].update(acc1.item(), n=batch_size)
            metric_loggers[i].meters['acc5'].update(acc5.item(), n=batch_size)

    ave = []
    # gather the stats from all processes
    for i in range(num_models):
        metric_loggers[i].synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_loggers[i].acc1, top5=metric_loggers[i].acc5, losses=metric_loggers[i].loss))

        ave.append({k: meter.global_avg for k, meter in metric_loggers[i].meters.items()})

    return ave
