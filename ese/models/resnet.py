# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# ionpy imports
from ionpy.nn.nonlinearity import get_nonlinearity
# misc imports
import ast
from typing import Any, Literal, Optional, Tuple


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels: int, 
        dims: int, 
        stride: int = 1, 
        use_norm: bool = False,
        activation: Optional[str] = "ReLU",
        downsample: Optional[Any] = None,
    ):
        super(BasicBlock, self).__init__()
        # Determine the classes we need to use based on the dimensionality of the input.
        conv = nn.Conv3d if dims == 3 else nn.Conv2d
        bn = nn.BatchNorm3d if dims == 3 else nn.BatchNorm2d

        # Get the nonlinearity we'll be using.
        self.nonlin_fn = get_nonlinearity(activation)()

        # Define the layers. (2 convs)
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_norm)
        self.bn1 = bn(out_channels) if use_norm else nn.Identity()

        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_norm)
        self.bn2 = bn(out_channels) if use_norm else nn.Identity()

        self.downsample = downsample
        self.use_norm = use_norm

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.nonlin_fn(out)

        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.nonlin_fn(out)

        return out


class SCTS(nn.Module):

    def __init__(
        self, 
        img_channels: int,
        num_classes: int,
        filters: list[int],
        dims: int = 2,
        use_norm: bool = False,
        use_image: bool = True,
        eps: float = 1e-4, 
        blocks_per_layer: int = 2,
        temp_range: Optional[Any] = None,
        conv_kws: Optional[dict[str, Any]] = {}
    ):
        super(SCTS, self).__init__()
        # Determine the classes we need to use based on the dimensionality of the input.
        self.conv_cls = nn.Conv3d if dims == 3 else nn.Conv2d
        self.bn_cls = nn.BatchNorm3d if dims == 3 else nn.BatchNorm2d

        self.eps = eps
        self.dims = dims
        self.conv_kws = conv_kws
        self.use_norm = use_norm
        self.use_image = use_image
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) if dims == 3 else nn.AdaptiveAvgPool2d((1, 1))
        # If we want to constrain the output.
        if isinstance(temp_range, str):
            temp_range = ast.literal_eval(temp_range)
        self.temp_range = temp_range

        # Define the first convolutional layer.
        self.conv1 = self.conv_cls(
            (num_classes + img_channels if use_image else num_classes), 
            filters[0], 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=not use_norm
        )
        self.bn1 = self.bn_cls(filters[0]) if use_norm else nn.Identity()
        self.act1 = get_nonlinearity(conv_kws.get("activation", "ReLU"))()

        # Define the set of blocks
        self.in_planes = filters[0]
        self.layer_dict = nn.ModuleDict()
        for i, f in enumerate(filters):
            if i == 0:
                self.layer_dict[f"layer_{i}"] = self._make_layer(f, blocks_per_layer)
            else:
                self.layer_dict[f"layer_{i}"] = self._make_layer(f, blocks_per_layer, stride=2)

        # The final fully connected layer.
        self.fc = nn.Linear(filters[-1], num_classes)

    def _make_layer(self, filters, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != filters:
            downsample = nn.Sequential(
                self.conv_cls(
                    self.in_planes, 
                    filters, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=not self.use_norm
                ),
                self.bn_cls(filters) if self.use_norm else nn.Identity(),
            )
        layers = []
        layers.append(
            BasicBlock(
                in_channels=self.in_planes, 
                out_channels=filters, 
                stride=stride, 
                dims=self.dims, 
                use_norm=self.use_norm,
                downsample=downsample,
                **self.conv_kws
            )
        )
        self.in_planes = filters 
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    in_channels=filters, 
                    out_channels=filters, 
                    dims=self.dims, 
                    use_norm=self.use_norm,
                    **self.conv_kws
                )
            )

        return nn.Sequential(*layers)

    def weights_init(self):
        pass

    def get_temp_map(self, logits, image):

        # Concatenate the image if we are using it.
        if self.use_image:
            x = torch.cat([logits, image], dim=1)
        else:
            x = logits 

        # Pass through the ResNet Block 
        x = self.act1(self.bn1(self.conv1(x)))
        if self.dims == 3:
            x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Go through the layers.
        for i in range(len(self.layer_dict)):
            x = self.layer_dict[f"layer_{i}"](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        unnorm_temp = self.fc(x)
        # We need to normalize our temperature to be positive.
        if self.temp_range is not None:
            temp = torch.sigmoid(unnorm_temp) * (self.temp_range[1] - self.temp_range[0]) + self.temp_range[0]
        else:
            temp = torch.abs(unnorm_temp) + self.eps

        print("Predicted Batch Temps: ", temp)
        # Repeat the temperature map for all classes.
        B = logits.shape[0]
        new_temp_map_shape = [B] + [1]*len(logits.shape[2:])
        expanded_temp_map = temp.view(new_temp_map_shape)
        # Reshape the temp map to match the logits.
        target_shape = [B] + list(logits.shape[2:])
        reshaped_temp_map = expanded_temp_map.expand(*target_shape)
        # Return the temp map.
        return reshaped_temp_map.unsqueeze(1) # Unsqueeze channel dim.

    def forward(self, logits, image=None, **kwargs):
        C = logits.shape[1] # B, C, spatial dims
        # Get the temperature map.
        temp_map = self.get_temp_map(logits, image) # B x 1 x spatial dims
        # Repeat the temperature map for all classes.
        rep_dims = [1, C] + [1] * (len(logits.shape) - 2)
        temp_map = temp_map.repeat(*rep_dims) # B x C x spatial dims
        # Assert that every position in the temp_map is positive.
        assert torch.all(temp_map >= 0), "Temperature map must be positive."
        # Finally, scale the logits by the temperatures.
        return logits / temp_map