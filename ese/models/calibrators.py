# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# misc imports
import math
from typing import Any, Optional
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
        

class TS(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        freeze_backbone: bool = True, 
        **kwargs
    ):
        super(TS, self).__init__()
        self.temp = nn.Parameter(torch.ones(1))
        self.backbone_model = model
        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

    def weights_init(self):
        self.temp.data.fill_(1)
    
    def get_temp_map(self, logits):
        all_dims = list(logits.shape)
        all_dims.pop(1) # remove the channel dimension
        return self.temp.repeat(*all_dims)

    def forward(self, x, **kwargs):
        # Pass through the backbone model.
        logits = self.backbone_model(x)
        # Get the temperature map.
        C = logits.shape[1]
        temp_map = self.get_temp_map(logits)
        # Expand the shape of temp map to match the shape of the logits
        num_spatial_dims = len(logits.shape) - 2 # Number of spatial dimensions 
        repeat_factors = [1, C] + [1] * num_spatial_dims
        repeated_temp_map = temp_map.unsqueeze(1).repeat(*repeat_factors)
        # Finally, scale the logits by the temperatures.
        tempered_logits = logits / repeated_temp_map
        # Return the tempered logits and the predicts temperatures.
        return tempered_logits

    @property
    def device(self):
        return next(self.parameters()).device


class LTS(nn.Module):
    def __init__(
        self, 
        model: nn.Module,
        img_channels: int,
        num_classes: int, 
        filters: list[int],
        use_image: bool = True,
        abs_output: bool = False,
        freeze_backbone: bool = True, 
        convs_per_block: int = 1,
        dims: int = 2,
        eps: float = 1e-6, 
        unet_conv_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        super(LTS, self).__init__()
        self.backbone_model = model
        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False
        self.calibrator_unet = UNet(
            in_channels=(num_classes + img_channels if use_image else num_classes), 
            out_channels=1, # For the temp map.
            filters=filters,
            dims=dims,
            convs_per_block=convs_per_block,
            conv_kws=unet_conv_kwargs
        )
        self.abs_output = abs_output
        self.use_image = use_image
        self.eps = eps 

    def weights_init(self):
        pass

    def get_temp_map(self, logits, image=None):
        # Concatenate the image if we are using it.
        if self.use_image:
            cal_input = torch.cat([logits, image], dim=1)
        else:
            cal_input = logits
        # Pass through the UNet.
        unnorm_temp_map = self.calibrator_unet(cal_input).squeeze(1) # B x Spat. Dims
        # Either we just abs value the output or we can do a complicated operation.
        if self.abs_output:
            temp_map = torch.abs(unnorm_temp_map)
        else:
            # Add ones so the temperature starts near 1.
            unnorm_temp_map += torch.ones(1, device=unnorm_temp_map.device)
            # Clip the values to be positive and add epsilon for smoothing.
            temp_map = F.relu(unnorm_temp_map)
        # Smooth with epsilon.
        temp_map += self.eps
        # Return the temp map.
        return temp_map

    def forward(self, x, image=None, **kwargs):
        # Pass through the backbone model.
        logits = self.backbone_model(x)
        # Get the temperature map.
        C = logits.shape[1]
        # Get the temperature map.
        temp_map = self.get_temp_map(logits, image) # B x Spatial Dims
        # Repeat the temperature map for all classes.
        num_spatial_dims = len(logits.shape) - 2 # Number of spatial dimensions 
        repeat_factors = [1, C] + [1] * num_spatial_dims
        repeated_temp_map = temp_map.unsqueeze(1).repeat(*repeat_factors) # B x C x H x W
        # Assert that every position in the temp_map is positive.
        assert torch.all(repeated_temp_map >= 0), "Temperature map must be positive."
        # Finally, scale the logits by the temperatures.
        tempered_logits = logits / repeated_temp_map
        # Return the tempered logits and the predicts temperatures.
        return tempered_logits

    @property
    def device(self):
        return next(self.parameters()).device


class VS(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int, 
        freeze_backbone: bool = True, 
        **kwargs
    ):
        super(VS, self).__init__()
        self.backbone_model = model
        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False
        self.vector_parameters = nn.Parameter(torch.ones(1, num_classes, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_classes, 1, 1))

    def weights_init(self):
        self.vector_parameters.data.fill_(1)
        self.vector_offset.data.fill_(0)

    def forward(self, x, **kwargs):
        # Pass through the backbone model.
        logits = self.backbone_model(x)
        # Scale the logits by the vector parameters and add the offset.
        return (self.vector_parameters * logits) + self.vector_offset

    @property
    def device(self):
        return next(self.parameters()).device


class DS(nn.Module):
    def __init__(
        self, 
        model: nn.Module,
        num_classes, 
        eps=1e-19, 
        freeze_backbone: bool = True, 
        **kwargs
    ):
        super(DS, self).__init__()
        self.backbone_model = model
        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False
        self.dirichlet_linear = nn.Linear(num_classes, num_classes)
        self.eps = eps

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, x, **kwargs):
        # Pass through the backbone model.
        logits = self.backbone_model(x)
        # Get the probabilities and log them.
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
        model: nn.Module,
        img_channels: int,
        num_classes: int, 
        use_image: bool = True,
        use_logits: bool = True, 
        freeze_backbone: bool = True, 
        eps=1e-12, 
        **kwargs
    ):
        super(IBTS, self).__init__()
        self.backbone_model = model
        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False
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
        # Get the temperature map.
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

    def forward(self, x, image, **kwargs):
        # Pass through the backbone model.
        logits = self.backbone_model(x)
        # Get the temperature map.
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
