# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# misc imports
import math
import pickle
# local imports
from ..metrics.global_ps import global_binwise_stats
from ..metrics.utils import (
    get_bins, 
    find_bins, 
    count_matching_neighbors
)
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
        

class Histogram_Binning(nn.Module):
    def __init__(
        self, 
        num_bins: int,
        num_classes: int,
        stats_file: str,
        normalize: bool, 
        **kwargs
    ):
        super(Histogram_Binning, self).__init__()
        # Load the data from the .pkl file
        with open(stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = global_binwise_stats(
            pixel_meters_dict=pixel_meters_dict["val"], # Use the validation set stats.
            num_bins=num_bins, # Use 15 bins
            num_classes=num_classes,
            class_conditioned=True,
            neighborhood_conditioned=False,
            class_wise=True,
            device="cuda"
        )
        self.val_freqs = gbs['bin_freqs'] # C x Bins
        # Get the bins and bin widths
        num_conf_bins = self.val_freqs.shape[1]
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_bins=num_conf_bins, 
            start=0.0,
            end=1.0
        )
        self.normalize = normalize

    def forward(self, logits, **kwargs):
        probs = torch.softmax(logits, dim=1) # B x C x H x W
        for lab_idx in range(probs.shape[1]):
            prob_map = probs[:, lab_idx, :, :] # B x H x W
            # Calculate the bin ownership map and transform the probs.
            prob_bin_ownership_map = find_bins(
                confidences=prob_map, 
                bin_starts=self.conf_bins,
                bin_widths=self.conf_bin_widths
            ).long() # B x H x W
            calibrated_prob_map = self.val_freqs[lab_idx][prob_bin_ownership_map] # B x H x W
            # Inserted the calibrated prob map back into the original prob map.
            probs[:, lab_idx, :, :] = calibrated_prob_map
        # If we are normalizing then we need to make sure the probabilities sum to 1.
        if self.normalize:
            probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    @property
    def device(self):
        return "cpu"


class NECTAR_Binning(nn.Module):
    def __init__(
        self, 
        num_bins: int,
        num_classes: int,
        neighborhood_width: int,
        stats_file: str,
        normalize: bool, 
        **kwargs
    ):
        super(NECTAR_Binning, self).__init__()
        # Load the data from the .pkl file
        with open(stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = global_binwise_stats(
            pixel_meters_dict=pixel_meters_dict["val"],
            num_bins=num_bins, # Use 15 bins
            num_classes=num_classes,
            neighborhood_width=neighborhood_width,       
            class_conditioned=True,
            neighborhood_conditioned=True,
            class_wise=True,
            device="cuda"
        )
        self.val_freqs = gbs['bin_freqs'] # C x Neighborhood Classes x Bins
        # Get the bins and bin widths
        num_conf_bins = self.val_freqs.shape[1]
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_bins=num_conf_bins, 
            start=0.0,
            end=1.0
        )
        self.normalize = normalize
        self.num_classes = num_classes
        self.neighborhood_width = neighborhood_width

    def forward(self, logits, **kwargs):
        probs = torch.softmax(logits, dim=1) # B x C x H x W
        hard_pred = torch.argmax(probs, dim=1) # B x H x W
        for lab_idx in range(self.num_classes):
            lab_prob_map = probs[:, lab_idx, :, :] # B x H x W
            lab_hard_pred = (hard_pred == lab_idx).long() # B x H x W
            # Calculate the bin ownership map and transform the probs.
            prob_bin_ownership_map = find_bins(
                confidences=lab_prob_map, 
                bin_starts=self.conf_bins,
                bin_widths=self.conf_bin_widths
            ) # B x H x W
            pred_num_neighb_map = count_matching_neighbors(
                lab_map=lab_hard_pred,
                neighborhood_width=self.neighborhood_width,
                binary=True
            ) # B x H x W
            calibrated_prob_map = torch.zeros_like(lab_prob_map)
            for nn_idx in range(self.neighborhood_width**2):
                neighbor_mask = (pred_num_neighb_map == nn_idx)
                # Replace the soft predictions with the old frequencies.
                calibrated_prob_map[neighbor_mask] =\
                    self.val_freqs[lab_idx][nn_idx][prob_bin_ownership_map][neighbor_mask].float()
            # Inserted the calibrated prob map back into the original prob map.
            probs[:, lab_idx, :, :] = calibrated_prob_map
        # If we are normalizing then we need to make sure the probabilities sum to 1.
        if self.normalize:
            probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    @property
    def device(self):
        return "cpu"


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


# NEighborhood-Conditional TemperAtuRe Scaling
class NECTAR_Scaling(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        neighborhood_width: int, 
        threshold: float = None, 
        eps: float = 1e-12,
        positive_constraint: bool = True,
        **kwargs
    ):
        super(NECTAR_Scaling, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.num_classes = num_classes
        self.neighborhood_width = neighborhood_width
        self.positive_constraint = positive_constraint
        # Define the parameters per neighborhood class
        num_neighbor_classes = neighborhood_width**2
        self.neighborhood_temps = nn.Parameter(torch.ones(num_neighbor_classes))

    def weights_init(self):
        self.neighborhood_temps.data.fill_(1)

    def forward(self, logits, **kwargs):
        # Softmax the logits to get probabilities
        probs = torch.softmax(logits, dim=1) # B C H W
        # Argnax over the channel dimension to get the current prediction
        if probs.shape[1] == 1:
            pred = (probs > self.threshold).float().squeeze(1) # B H W
        else:
            pred = torch.argmax(probs, dim=1) # B H W
        # Get the per-pixel num neighborhood class
        pred_matching_neighbors_map = count_matching_neighbors(
            lab_map=pred, 
            neighborhood_width=self.neighborhood_width
        ).unsqueeze(1) # B 1 H W
        # Place the temperatures in the correct positions
        neighborhood_temp_map = self.neighborhood_temps[pred_matching_neighbors_map]
        # Apply this to all classes.
        temps = neighborhood_temp_map.repeat(1, self.num_classes, 1, 1) # B C H W
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
    def __init__(self, num_classes, image_channels, eps=1e-8, **kwargs):
        super(LTS, self).__init__()
        self.num_classes = num_classes

        self.temperature_level_2_conv1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

        self.temperature_level_2_param1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

        self.temperature_level_2_conv_img = nn.Conv2d(image_channels, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

        self.eps = eps 

    def weights_init(self):
        pass

    def forward(self, logits, image, label=None):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda()
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda()
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda()
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda()
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)

        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_num_class = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))

        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_num_class * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + torch.ones(1).cuda()

        temp_param = self.temperature_level_2_param_img(logits)

        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        temperature = F.relu(temperature + torch.ones(1).cuda()) + self.eps # + 1 is done because the temperature starts near 0.
        temperature = temperature.repeat(1, self.num_classes, 1, 1)
        return logits / temperature

    @property
    def device(self):
        return next(self.parameters()).device


class Selective_Scaling(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(Selective_Scaling, self).__init__()
        self.dirichlet_linear = nn.Linear(num_classes, num_classes)
        self.binary_linear = nn.Linear(num_classes, 2)
        
        self.bn0 = nn.BatchNorm2d(num_classes)
        self.linear_1 = nn.Linear(num_classes, num_classes*2)
        self.bn1 = nn.BatchNorm2d(num_classes*2)
        self.linear_2 = nn.Linear(num_classes*2, num_classes)
        self.bn2 = nn.BatchNorm2d(num_classes)

        self.relu = nn.ReLU()        

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, logits, images=None, labels=None):
        logits = logits.permute(0,2,3,1)   
        probs = torch.softmax(logits)
        ln_probs = torch.log(probs + 1e-16)

        out = self.dirichlet_linear(ln_probs)
        out = self.bn0(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_1(out.permute(0,2,3,1))
        out = self.bn1(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_2(out.permute(0,2,3,1))
        out = self.bn2(out.permute(0,3,1,2))
        out = self.relu(out)       

        tf_positive = self.binary_linear(out.permute(0,2,3,1))
        
        return  tf_positive.permute(0,3,1,2)

    @property
    def device(self):
        return next(self.parameters()).device