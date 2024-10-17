# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# ionpy imports
from ionpy.nn.nonlinearity import get_nonlinearity
# local imports
from .utils import get_temp_map
# misc imports
import ast
from typing import Any, List, Optional


class FeatureRegressor(nn.Module):

    def __init__(
        self, 
        in_features: int,
        num_classes: int,
        dims: int,
        features: List[int],
        backbone_model: nn.Module,
        pool_fn: str = "mean",
        activation: str = "LeakyReLU",
    ):
        super(FeatureRegressor, self).__init__()
        # We use the backbone to get an encoded representation of the input.
        self.backbone = backbone_model

        # If pool_func is 'max' then use max pooling, otherwise use average pooling.
        if pool_fn == 'max':
            self.pool = nn.AdaptiveMaxPool3d((1, 1, 1)) if dims == 3 else nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if dims == 3 else nn.AdaptiveAvgPool2d((1, 1))
        
        # Then we pass through some number of fully connected layers.
        # For this to be tracked by PyTorch we need to use a module dict.
        self.fc_layers = nn.ModuleDict()
        self.activation = get_nonlinearity(activation)()
        self.num_layers = len(features)
        for i, f in enumerate(features):
            if i == 0:
                self.fc_layers[f"fc_{i}"] = nn.Linear(in_features, f)
            else:
                self.fc_layers[f"fc_{i}"] = nn.Linear(features[i - 1], f)
        
        # Define the output layer.
        self.out_linear = nn.Linear(features[-1], num_classes)

    def forward(self, image, **kwargs):

        _, x_feats = self.backbone.encode(image)

        # Do a global pool over the features.
        pooled_feats = self.pool(x_feats)
        x = pooled_feats.view(pooled_feats.size(0), -1)

        # Pass through the fully connected layers.
        for i in range(self.num_layers):
            x = self.fc_layers[f"fc_{i}"](x)
            x = self.activation(x)
        
        # Finally pass through the output layer.
        out = self.out_linear(x)

        # Return the output of the fc layer.
        return out 



class E2T(nn.Module):

    def __init__(
        self, 
        in_features: int,
        num_classes: int,
        dims: int,
        features: List[int],
        backbone_model: nn.Module,
        eps: float = 1e-6, 
        pool_fn: str = "mean",
        activation: str = "LeakyReLU",
        temp_range: Optional[Any] = None,
    ):
        super(E2T, self).__init__()
        # We use the backbone to get an encoded representation of the input.
        self.regressor = FeatureRegressor(
            in_features=in_features,
            num_classes=num_classes,
            dims=dims,
            features=features,
            backbone_model=backbone_model,
            pool_fn=pool_fn,
            activation=activation
        ) 
        self.eps = eps
        # If we want to constrain the output.
        if isinstance(temp_range, str):
            temp_range = ast.literal_eval(temp_range)
        self.temp_range = temp_range

    def weights_init(self):
        pass

    def pred_temps(self, image):

        # Pass through the pretrained backbone + fc layers.
        unnorm_temp = self.regressor(image)

        # We need to normalize our temperature to be positive.
        if self.temp_range is not None:
            temps = torch.sigmoid(unnorm_temp) * (self.temp_range[1] - self.temp_range[0]) + self.temp_range[0]
        else:
            temps = torch.abs(unnorm_temp) # If we don't have a range, need to at least set to [0, inf)

        # Add eps as a precaution so we don't div by zero.
        smoothed_temps = temps + self.eps
        print(smoothed_temps)

        # Return the temperature map.
        return smoothed_temps
    

    def forward(self, logits, image, **kwargs):
        # First we predict the temperatures.
        temps = self.pred_temps(image) # B 

        # Then use the predicted temperatures to get the temperature map.
        temp_map = get_temp_map(temps, pred_shape=logits.shape) # B x 1 x spatial dims

        # Return the tempered logits and the predicted temperatures.
        return logits / temp_map