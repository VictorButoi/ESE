# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# misc imports
import math
from typing import Any, Literal, Optional
# local imports
from .unet import UNet
from .utils import create_gaussian_tensor
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
        all_dims = list(logits.shape)
        all_dims.pop(1) # remove the channel dimension
        return self.temp.repeat(*all_dims)

    def forward(self, logits, **kwargs):
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


class Popcorn_Scaling(nn.Module):
    def __init__(
        self, 
        ksize: int, 
        init_mode: Literal['delta', 'uniform', 'gaussian'] = 'delta',
        **kwargs
    ):
        raise NotImplementedError("Popcorn_Scaling is not implemented yet.")

    def weights_init(self):
        raise NotImplementedError("Popcorn_Scaling is not implemented yet.")
        
    def forward(self, logits, **kwargs):
        raise NotImplementedError("Popcorn_Scaling is not implemented yet.")

    @property
    def device(self):
        raise NotImplementedError("Popcorn_Scaling is not implemented yet.")


class LocalTS(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_classes: int, 
        filters: list[int],
        use_image: bool = True,
        abs_output: bool = False,
        convs_per_block: int = 1,
        dims: int = 2,
        eps: float = 1e-6, 
        unet_conv_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        super(LocalTS, self).__init__()
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

    def forward(self, logits, image=None, **kwargs):
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
