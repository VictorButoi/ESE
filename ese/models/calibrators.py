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
        temp_map = temp_map.unsqueeze(1).repeat(*repeat_factors)
        # Finally, scale the logits by the temperatures.
        return logits / temp_map

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
        super(Popcorn_Scaling, self).__init__()
        assert ksize % 2 == 1, "Kernel size must be odd."
        self.ksize = ksize
        self.init_mode = init_mode
        self.temperature_kernel = nn.Parameter(torch.ones(ksize, ksize))

    def weights_init(self):
        if self.init_mode == 'random':
            # Randomly initialize the kernel weights
            self.temperature_kernel.data.normal_(0, 0.01)
        elif self.init_mode == 'delta':
            self.temperature_kernel.data.fill_(0)
            self.temperature_kernel.data[self.ksize//2, self.ksize//2] = 1
        elif self.init_mode == 'uniform':
            # Initialize the kernel as uniform with each element being 1/ksize^2
            self.temperature_kernel.data.fill_(1 / (self.ksize**2))
        elif self.init_mode == 'gaussian':
            # Initialize the kernel as a guassian centered at the middle pixel.
            self.temperature_kernel = nn.Parameter(create_gaussian_tensor(mu=0.0, sigma=1.0, ksize=self.ksize))
        else:
            raise ValueError(f"Invalid init_mode: {self.init_mode}")
        
    def forward(self, logits, **kwargs):
        # Expand the shape of the kernel to match the shape of the logits.
        temperature_kernel = self.temperature_kernel[None, None, ...]
        # Convolve the logits with the temperature kernel.
        tempered_logits = F.conv2d(logits, temperature_kernel, padding=self.ksize//2) 
        # Finally, scale the logits by the temperatures.
        return tempered_logits

    @property
    def device(self):
        return next(self.parameters()).device


class Gaussian_Temperature_Scaling(nn.Module):
    def __init__(self, ksize, **kwargs):
        super(Gaussian_Temperature_Scaling, self).__init__()
        assert ksize % 2 == 1, "Kernel size must be odd."
        # Initialize the kernel with a Gaussian distribution.
        self.mu = nn.Parameter(torch.ones(1))
        self.sigma = nn.Parameter(torch.zeros(1))
        self.ksize = ksize
    
    def forward(self, logits, **kwargs):
        gaussian_temp_kernel = create_gaussian_tensor(
            mu=self.mu,
            sigma=self.sigma,
            ksize=self.ksize 
        )[None, None, ...]
        # Expand the shape of temp map to match the shape of the logits
        tempered_logits = F.conv2d(logits, gaussian_temp_kernel, padding=self.ksize//2) 
        # Finally, scale the logits by the temperatures.
        return tempered_logits

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


class ImageBasedTS(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_classes: int, 
        use_image: bool = True,
        use_logits: bool = True, 
        eps=1e-12, 
        **kwargs
    ):
        super(ImageBasedTS, self).__init__()

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
        # Assert that every position in the temp_map is positive.
        assert torch.all(temp_map >= 0), "Temperature map must be positive."
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
        dims: int = 2,
        eps=1e-12, 
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
        unnorm_temp_map = self.calibrator_unet(cal_input).squeeze(1) # B x Spat. Dims
        # Add ones so the temperature starts near 1.
        unnorm_temp_map += torch.ones(1, device=unnorm_temp_map.device)
        # Clip the values to be positive and add epsilon for smoothing.
        temp_map = F.relu(unnorm_temp_map) + self.eps
        # Return the temp map.
        return temp_map

    def forward(self, logits, image, **kwargs):
        C = logits.shape[1]
        # Get the temperature map.
        temp_map = self.get_temp_map(logits, image) # B x Spatial Dims
        # Repeat the temperature map for all classes.
        num_spatial_dims = len(logits.shape) - 2 # Number of spatial dimensions 
        repeat_factors = [1, C] + [1] * num_spatial_dims
        temp_map = temp_map.unsqueeze(1).repeat(*repeat_factors) # B x C x H x W
        # Assert that every position in the temp_map is positive.
        assert torch.all(temp_map >= 0), "Temperature map must be positive."
        # Finally, scale the logits by the temperatures.
        return logits / temp_map

    @property
    def device(self):
        return next(self.parameters()).device
