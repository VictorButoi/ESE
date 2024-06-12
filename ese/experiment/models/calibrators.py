# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# misc imports
import math
# local imports
from .unet import UNet
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
    
    def get_temp_map(self, logits):
        B, _, H, W = logits.shape
        temp_map = self.temp.repeat(B, H, W)
        return temp_map

    def forward(self, logits, **kwargs):
        _, C, _, _ = logits.shape
        temp_map = self.get_temp_map(logits)
        # Expand the shape of temp map to match the shape of the logits
        temp_map = temp_map.unsqueeze(1).repeat(1, C, 1, 1)
        # Finally, scale the logits by the temperatures.
        return logits / temp_map

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


class IBTS(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_classes: int, 
        use_image: bool = True,
        use_logits: bool = True, 
        eps=1e-12, 
        **kwargs
    ):
        super(IBTS, self).__init__()

        self.in_conv = nn.Conv2d(in_channels=(num_classes + img_channels if use_image else num_classes), out_channels=3, kernel_size=1)
        self.temp_predictor = resnet18()
        # Replace the last fully-connected layer to output a single number.
        self.temp_predictor.fc = nn.Linear(512, 1)
        # Track some info about this calibrator.
        self.use_image = use_image
        self.use_logits = use_logits
        self.eps = eps 

    def weights_init(self):
        pass

    def get_temp_map(self, logits, image):
        _, _, H, W = logits.shape
        # Either passing into probs or logits into UNet, can affect optimization.
        if not self.use_logits:
            cal_input = torch.softmax(logits, dim=1)
        else:
            cal_input = logits
        # Concatenate the image if we are using it.
        if self.use_image:
            cal_input = torch.cat([cal_input, image], dim=1)
        # Pass through the in conv
        x = self.in_conv(cal_input)
        # Pass through the UNet
        unnorm_temp = self.temp_predictor(x) # B x 1
        # Add ones so the temperature starts near 1.
        unnorm_temp += torch.ones(1, device=unnorm_temp.device)
        # Finally normalize it to be positive and add epsilon for smoothing.
        temp = F.relu(unnorm_temp) + self.eps
        # Clip the values to be positive and add epsilon for smoothing.
        temp_map = temp.unsqueeze(1).repeat(1, H, W)
        # Return the temp map.
        return temp_map

    def forward(self, logits, image, **kwargs):
        _, C, _, _ = logits.shape
        # Get the temperature map.
        temp_map = self.get_temp_map(logits, image) # B x H x W
        # Repeat the temperature map for all classes.
        temp_map = temp_map.unsqueeze(1).repeat(1, C, 1, 1) # B x C x H x W
        # Finally, scale the logits by the temperatures.
        return logits / temp_map

    @property
    def device(self):
        return next(self.parameters()).device

        
class LocalTS(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_classes: int, 
        filters: list[int],
        use_image: bool = True,
        use_logits: bool = True, 
        convs_per_block: int = 1,
        eps=1e-12, 
        **kwargs
    ):
        super(LocalTS, self).__init__()

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

    def get_temp_map(self, logits, image):
        # Either passing into probs or logits into UNet, can affect optimization.
        if not self.use_logits:
            cal_input = torch.softmax(logits, dim=1)
        else:
            cal_input = logits
        # Concatenate the image if we are using it.
        if self.use_image:
            cal_input = torch.cat([cal_input, image], dim=1)
        # Pass through the UNet.
        unnorm_temp_map = self.calibrator_unet(cal_input).squeeze(1) # B x H x W
        # Add ones so the temperature starts near 1.
        unnorm_temp_map += torch.ones(1, device=unnorm_temp_map.device)
        # Clip the values to be positive and add epsilon for smoothing.
        temp_map = F.relu(unnorm_temp_map) + self.eps
        # Return the temp map.
        return temp_map

    def forward(self, logits, image, **kwargs):
        _, C, _, _ = logits.shape
        # Get the temperature map.
        temp_map = self.get_temp_map(logits, image) # B x H x W
        # Repeat the temperature map for all classes.
        temp_map = temp_map.unsqueeze(1).repeat(1, C, 1, 1) # B x C x H x W
        # Finally, scale the logits by the temperatures.
        return logits / temp_map

    @property
    def device(self):
        return next(self.parameters()).device


# Keeping this for running older models.
class LTS(nn.Module):
    def __init__(self, num_classes, image_channels, sigma=1e-8, **kwargs):
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

        self.sigma = sigma

    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

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
        temperature = F.relu(temperature + torch.ones(1).cuda()) + self.sigma 
        temperature = temperature.repeat(1, self.num_classes, 1, 1)
        return logits / temperature