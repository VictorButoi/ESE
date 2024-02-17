# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# misc imports
import math
# local imports
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
