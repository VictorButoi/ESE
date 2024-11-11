# torch imports
import torch.nn as nn
from ionpy.experiment.util import eval_config


# Define a combined loss function that sums individual losses
class CombinedLoss(nn.Module):
    def __init__(self, loss_func_dict, loss_func_weights):
        super(CombinedLoss, self).__init__()
        self.loss_fn_dict = nn.ModuleDict(loss_func_dict)
        self.loss_func_weights = loss_func_weights
    def forward(self, outputs, targets):
        total_loss = 0
        for loss_name, loss_func in self.loss_fn_dict.items():
            total_loss += self.loss_func_weights[loss_name] * loss_func(outputs, targets)
        return total_loss


def eval_combo_config(loss_config):
    # Combined loss functions case
    combo_losses = loss_config["_combo_class"]
    # Instantiate each loss function using eval_config
    loss_fn_dict = {} 
    loss_fn_weights = {} 
    for name, config in combo_losses.items():
        cfg_dict = config.to_dict()
        loss_fn_weights[name] = cfg_dict.pop("weight", 1.0)
        loss_fn_dict[name] = eval_config(cfg_dict)
    return CombinedLoss(
        loss_func_dict=loss_fn_dict,
        loss_func_weights=loss_fn_weights
    )