# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# misc imports
import math
import matplotlib.pyplot as plt
# local imports
from .unet import UNet
from ..metrics.utils import agg_neighbors_preds
# Set the print options
torch.set_printoptions(sci_mode=False, precision=3)
    

def initialization(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        

class Temperature_Scaling(nn.Module):
    def __init__(self, **kwargs):
        super(Temperature_Scaling, self).__init__()
        self.temp = nn.Parameter(torch.ones(1))

    def weights_init(self):
        self.temp.data.fill_(1)

    def forward(self, logits, **kwargs):
        return logits / self.temp 

    @property
    def device(self):
        return next(self.parameters()).device


class Vector_Scaling(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(Vector_Scaling, self).__init__()
        self.vector_parameters = nn.Parameter(torch.ones(1, num_classes, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_classes, 1, 1))

    def weights_init(self):
        self.vector_parameters.data.fill_(1)
        self.vector_offset.data.fill_(0)

    def forward(self, logits, **kwargs):
        return (self.vector_parameters * logits) + self.vector_offset

    @property
    def device(self):
        return next(self.parameters()).device


class Dirichlet_Scaling(nn.Module):
    def __init__(self, num_classes, eps=1e-19, **kwargs):
        super(Dirichlet_Scaling, self).__init__()
        self.dirichlet_linear = nn.Linear(num_classes, num_classes)
        self.eps = eps

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, logits, **kwargs):
        probs = torch.softmax(logits, dim=1)
        ln_probs = torch.log(probs + self.eps) # B x C x H x W
        # Move channel dim to the back (for broadcasting)
        ln_probs = ln_probs.permute(0,2,3,1).contiguous() # B x H x W x C
        ds_probs = self.dirichlet_linear(ln_probs)
        ds_probs = ds_probs.permute(0,3,1,2).contiguous() # B x C x H x W
        # Return scaled log probabilities
        return ds_probs

    @property
    def device(self):
        return next(self.parameters()).device

        
class LTS(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_classes: int, 
        filters: list[int],
        use_image: bool,
        use_logits: bool, 
        convs_per_block: int = 1,
        eps=1e-12, 
        **kwargs
    ):
        super(LTS, self).__init__()

        self.num_classes = num_classes
        self.calibrator_unet = UNet(
            in_channels=(num_classes + img_channels if use_image else num_classes), 
            out_channels=1, # For the temp map.
            filters=filters,
            convs_per_block=convs_per_block,
        )
        self.use_image = use_image
        self.use_logits = use_logits
        self.eps = eps 

    def weights_init(self):
        pass

    def forward(self, logits, image, **kwargs):
        # Either passing into probs or logits into UNet, can affect optimization.
        if not self.use_logits:
            cal_input = torch.softmax(logits, dim=1)
        else:
            cal_input = logits
        # Concatenate the image if we are using it.
        if self.use_image:
            cal_input = torch.cat([cal_input, image], dim=1)
        # Pass through the UNet.
        unnorm_temp_map = self.calibrator_unet(cal_input)
        # Add ones so the temperature starts near 1.
        unnorm_temp_map += torch.ones(1, device=unnorm_temp_map.device)
        # Clip the values to be positive and add epsilon for smoothing.
        temp_map = F.relu(unnorm_temp_map) + self.eps
        # Repeat the temperature map for all classes.
        temp_map = temp_map.repeat(1, self.num_classes, 1, 1)
        # Finally, scale the logits by the temperatures.
        return logits / temp_map

    @property
    def device(self):
        return next(self.parameters()).device


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
