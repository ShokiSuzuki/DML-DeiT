import torch
from torch.nn import functional as F
import json
import utils



class DMLLoss(torch.nn.Module):

    def __init__(self, base_criterion: torch.nn.Module,
                distillation_type: str, alpha: float, tau: float, output_dir: str):
        super().__init__()
        self.base_criterion = base_criterion
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.output_dir = output_dir

    def forward(self, model_id, outputs, labels, output_loss: bool=False):

        T = self.tau
        for i in range(len(outputs)):
            if i == model_id:
                continue
            if not isinstance(outputs[i], torch.Tensor):
                output, output_kd = outputs[i]
                outputs[i] = (output + output_kd) / 2

        outputs1    = outputs[model_id]
        outputs1_kd = None
        if not isinstance(outputs[model_id], torch.Tensor):
            outputs1, outputs1_kd = outputs[model_id]

        base_loss = self.base_criterion(outputs1, labels)
        if self.distillation_type == 'none' or len(outputs) == 1:
            return base_loss

        distillation_loss = 0.0
        if self.distillation_type == 'soft':
            if outputs1_kd is not None:
                for i in range(len(outputs)):
                    if i == model_id:
                        continue
                    distillation_loss += F.kl_div(
                        F.log_softmax(outputs1_kd / T, dim=1),
                        F.log_softmax(outputs[i].detach() / T, dim=1),
                        reduction='batchmean',
                        log_target=True
                    ) * (T * T)
            else:
                for i in range(len(outputs)):
                    if i == model_id:
                        continue
                    distillation_loss += F.kl_div(
                        F.log_softmax(outputs1 / T, dim=1),
                        F.log_softmax(outputs[i].detach() / T, dim=1),
                        reduction='batchmean',
                        log_target=True
                    ) * (T * T)
        else:
            if outputs1_kd is not None:
                for i in range(len(outputs)):
                    if i == model_id:
                        continue
                    distillation_loss += F.cross_entropy(outputs1_kd, outputs[i].argmax(dim=1))
            else:
                for i in range(len(outputs)):
                    if i == model_id:
                        continue
                    distillation_loss += F.cross_entropy(outputs1, outputs[i].argmax(dim=1))
        
        distillation_loss = distillation_loss / (len(outputs) - 1)
        loss = base_loss + distillation_loss
            
        if output_loss:
            log_stats = {'base_loss': base_loss.item(), 'distillation_loss': distillation_loss.item()}

            if self.output_dir and utils.is_main_process():
                with (self.output_dir / "loss.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        return loss

