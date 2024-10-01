# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# misc imports
from typing import Any, Literal, Optional


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels: int, 
        dims: int, 
        stride: int = 1, 
        use_norm: bool = False,
        downsample: Optional[Any] = None,
    ):
        super(BasicBlock, self).__init__()
        # Determine the classes we need to use based on the dimensionality of the input.
        conv = nn.Conv3d if dims == 3 else nn.Conv2d
        bn = nn.BatchNorm3d if dims == 3 else nn.BatchNorm2d
        # Define the layers. (2 convs)
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_norm)
        self.bn1 = bn(out_channels) if use_norm else nn.Identity()

        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_norm)
        self.bn2 = bn(out_channels) if use_norm else nn.Identity()

        self.downsample = downsample
        self.use_norm = use_norm

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out


class SubConTS(nn.Module):

    def __init__(
        self, 
        img_channels: int,
        num_classes: int,
        filters: list[int],
        dims: int = 2,
        eps: float = 1e-12,
        use_norm: bool = False,
        use_image: bool = True,
        blocks_per_layer: int = 2,
        **kwargs
    ):
        super(SubConTS, self).__init__()
        # Determine the classes we need to use based on the dimensionality of the input.
        self.conv_cls = nn.Conv3d if dims == 3 else nn.Conv2d
        self.bn_cls = nn.BatchNorm3d if dims == 3 else nn.BatchNorm2d

        self.eps = eps
        self.dims = dims
        self.use_norm = use_norm
        self.use_image = use_image
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) if dims == 3 else nn.AdaptiveAvgPool2d((1, 1))

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

        # Define the set of blocks
        self.in_planes = filters[0]
        self.layer1 = self._make_layer(filters=filters[0], num_blocks=blocks_per_layer)
        self.layer2 = self._make_layer(filters=filters[1], num_blocks=blocks_per_layer, stride=2)
        self.layer3 = self._make_layer(filters=filters[2], num_blocks=blocks_per_layer, stride=2)
        self.layer4 = self._make_layer(filters=filters[3], num_blocks=blocks_per_layer, stride=2)

        # The final fully connected layer.
        self.fc = nn.Linear(512, num_classes)

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
                downsample=downsample
            )
        )
        self.in_planes = filters 
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    in_channels=filters, 
                    out_channels=filters, 
                    dims=self.dims, 
                    use_norm=self.use_norm
                )
            )

        return nn.Sequential(*layers)

    def weights_init(self):
        pass

    def get_temp_map(self, logits, image):
        # Concatenate the image if we are using it.
        if self.use_image:
            cal_input = torch.cat([logits, image], dim=1)
        else:
            cal_input = logits 
        print(cal_input.shape)
        # Pass through the ResNet Blocks 
        x = F.relu(self.bn1(self.conv1(cal_input)))
        print(x.shape)
        if self.dims == 3:
            x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        unnorm_temp = self.fc(x)
        print(unnorm_temp.shape)
        # Add ones so the temperature starts near 1.
        unnorm_temp += torch.ones(1, device=unnorm_temp.device)
        # Finally normalize it to be positive and add epsilon for smoothing.
        temp = F.relu(unnorm_temp) + self.eps
        # Clip the values to be positive and add epsilon for smoothing.
        rep_dims = [1] + list(logits.shape[2:])
        temp_map = temp.unsqueeze(1).repeat(*rep_dims)
        # Return the temp map.
        return temp_map

    def forward(self, logits, image=None, **kwargs):
        C = logits.shape[1] # B, C, spatial dims
        # Get the temperature map.
        temp_map = self.get_temp_map(logits, image) # B x spatial dims
        # Repeat the temperature map for all classes.
        rep_dims = [1, C] + [1] * (len(logits.shape) - 2)
        temp_map = temp_map.unsqueeze(1).repeat(*rep_dims) # B x C x spatial dims
        # Assert that every position in the temp_map is positive.
        assert torch.all(temp_map >= 0), "Temperature map must be positive."
        # Finally, scale the logits by the temperatures.
        return logits / temp_map