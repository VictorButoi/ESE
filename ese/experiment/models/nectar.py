# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# misc imports
import math
import matplotlib.pyplot as plt
# local imports
from .unet import UNet
from ..metrics.utils import agg_neighbors_preds
# Set the print options
torch.set_printoptions(sci_mode=False, precision=3)


# NEighborhood-Conditional TemperAtuRe Scaling
class NECTAR_Scaling(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        neighborhood_width: int, 
        class_wise: bool = False,
        eps: float = 1e-12,
        threshold: float = 0.5, 
        positive_constraint: bool = True,
        **kwargs
    ):
        super(NECTAR_Scaling, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.class_wise = class_wise
        self.num_classes = num_classes
        self.neighborhood_width = neighborhood_width
        self.positive_constraint = positive_constraint
        # Define the parameters per neighborhood class
        num_neighbor_classes = neighborhood_width**2
        if class_wise:
            self.class_wise_nt = nn.Parameter(torch.ones(num_classes, num_neighbor_classes))
        else:
            self.neighborhood_temps = nn.Parameter(torch.ones(num_neighbor_classes))

    def weights_init(self):
        if self.class_wise:
            self.class_wise_nt.data.fill_(1)
        else:
            self.neighborhood_temps.data.fill_(1)

    def forward(self, logits, **kwargs):
        # Softmax the logits to get probabilities
        y_probs = torch.softmax(logits, dim=1) # B C H W
        # Argnax over the channel dimension to get the current prediction
        if y_probs.shape[1] == 1:
            y_hard = (y_probs > self.threshold).float().squeeze(1) # B H W
        else:
            y_hard = torch.argmax(y_probs, dim=1) # B H W
        # Get the per-pixel num neighborhood class
        neighbor_agg_map = agg_neighbors_preds(
            pred_map=y_hard, 
            neighborhood_width=self.neighborhood_width,
            class_wise=False,
            discrete=True,
            binary=False
        ) # B 1 H W
        if self.class_wise:
            # Place the temperatures in the correct positions
            neighborhood_temp_map = torch.zeros_like(y_probs)
            for class_idx in range(self.num_classes):
                # Get the mask for the current class
                neighborhood_temp_map[y_hard==class_idx] = self.class_wise_nt[class_idx][neighbor_agg_map][y_hard==class_idx]
        else:
            # Place the temperatures in the correct positions
            neighborhood_temp_map = self.neighborhood_temps[neighbor_agg_map]
        # Apply this to all classes.
        temps = neighborhood_temp_map.unsqueeze(1).repeat(1, self.num_classes, 1, 1) # B C H W
        # If we are constrained to have a positive temperature, then we need to guide
        # the optimization to picking a parameterization that is positive.
        if self.positive_constraint:
            temps = F.relu(temps)
        # Add an epsilon to avoid dividing by zero
        temps = temps + self.eps
        # Finally, scale the logits by the temperatures
        return logits / temps 

    @property
    def device(self):
        return next(self.parameters()).device

# NEighborhood-Conditional TemperAtuRe Scaling
class NS_V2(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        neighborhood_width: int, 
        eps: float = 1e-12,
        threshold: float = 0.5, 
        positive_constraint: bool = True,
        **kwargs
    ):
        super(NS_V2, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.num_classes = num_classes
        self.neighborhood_width = neighborhood_width
        self.positive_constraint = positive_constraint
        # Define the parameters per neighborhood class
        num_neighbor_classes = neighborhood_width**2
        self.class_wise_nt = nn.Parameter(torch.ones(num_classes, num_neighbor_classes))

    def weights_init(self):
        self.class_wise_nt.data.fill_(1)

    def forward(self, logits, **kwargs):
        # Softmax the logits to get probabilities
        y_probs = torch.softmax(logits, dim=1) # B C H W
        # Argnax over the channel dimension to get the current prediction
        if y_probs.shape[1] == 1:
            y_hard = (y_probs > self.threshold).float().squeeze(1) # B H W
        else:
            y_hard = torch.argmax(y_probs, dim=1) # B H W
        # Get the per-pixel num neighborhood class
        neighbor_agg_map = agg_neighbors_preds(
            pred_map=y_hard, 
            neighborhood_width=self.neighborhood_width,
            class_wise=False,
            discrete=True,
            binary=False
        ) # B 1 H W
        # Place the temperatures in the correct positions
        neighborhood_temp_map = torch.zeros((y_hard.shape), device=y_probs.device, dtype=y_probs.dtype)
        for class_idx in range(self.num_classes):
            # Get the mask for the current class
            neighborhood_temp_map[y_hard==class_idx] = self.class_wise_nt[class_idx][neighbor_agg_map][y_hard==class_idx]
        # Apply this to all classes.
        temps = neighborhood_temp_map.unsqueeze(1).repeat(1, self.num_classes, 1, 1) # B x C x H x W
        # If we are constrained to have a positive temperature, then we need to guide
        # the optimization to picking a parameterization that is positive.
        if self.positive_constraint:
            temps = F.relu(temps)
        # Add an epsilon to avoid dividing by zero
        temps = temps + self.eps
        # Finally, scale the logits by the temperatures
        return logits / temps 

    @property
    def device(self):
        return next(self.parameters()).device