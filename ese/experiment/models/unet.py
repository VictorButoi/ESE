from dataclasses import dataclass
from typing import Optional, Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from IonPy.nn import get_nonlinearity, ConvBlock
from pydantic import validate_arguments


@validate_arguments
@dataclass(eq=False, repr=False)
class UNet(nn.Module):

    in_channels: int
    out_channels: int
    filters: list[int]
    dec_filters: Optional[list[int]] = None
    out_activation: Optional[str] = None
    convs_per_block: int = 1
    skip_connections: bool = True
    dims: int = 2
    interpolation_mode: str = "linear"
    conv_kws: Optional[dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        filters = list(self.filters)
        if self.dec_filters is None:
            dec_filters = filters[-2::-1]
        else:
            dec_filters = list(self.dec_filters)
        assert len(dec_filters) == len(filters) - 1

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        conv_kws = self.conv_kws or {}

        for in_ch, out_ch in zip([self.in_channels] + filters[:-1], filters):
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_kws)
            self.enc_blocks.append(c)

        prev_out_ch = filters[-1]
        skip_chs = filters[-2::-1]
        for skip_ch, out_ch in zip(skip_chs, dec_filters):
            in_ch = skip_ch + prev_out_ch if self.skip_connections else prev_out_ch
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_kws)
            prev_out_ch = out_ch
            self.dec_blocks.append(c)

        # Use convblock to benefit from .reset_parameters
        self.out_conv = ConvBlock(
            prev_out_ch,
            [self.out_channels],
            activation=None,
            kernel_size=1,
            dims=self.dims,
            norm=None,
            residual=False,
        )

        if self.interpolation_mode == "linear":
            self.interpolation_mode = ["linear", "bilinear", "trilinear"][self.dims - 1]

        if self.out_activation:
            if self.out_activation == "Softmax":
                self.out_fn = nn.Softmax(dim=1)
            else:
                self.out_fn = get_nonlinearity(self.out_activation)()

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:

        conv_outputs = []

        pool_fn = getattr(F, f"max_pool{self.dims}d")

        for i, conv_block in enumerate(self.enc_blocks):
            x = conv_block(x)
            if i == len(self.enc_blocks) - 1:
                break
            conv_outputs.append(x)
            x = pool_fn(x, 2)

        for i, conv_block in enumerate(self.dec_blocks, start=1):
            x = F.interpolate(
                x,
                size=conv_outputs[-i].size()[-self.dims :],
                align_corners=True,
                mode=self.interpolation_mode,
            )
            if self.skip_connections:
                x = torch.cat([x, conv_outputs[-i]], dim=1)
            x = conv_block(x)

        x = self.out_conv(x)

        if self.out_activation:
            x = self.out_fn(x)

        return x
