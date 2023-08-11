from dataclasses import dataclass
import equinox as eqx
import jax
from pydantic import validate_arguments
from typing import Any, Optional
from .modules import ConvBlock


@validate_arguments
@dataclass(eq=False, repr=False)
class UNet(eqx.Module):

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
        
        self.enc_blocks = eqx.ModuleList()
        self.dec_blocks = eqx.ModuleList()

        for in_ch, out_ch in zip([self.in_channels] + filters[:-1], filters):
            c = eqx.ConvBlock(in_ch, [out_ch] * self.convs_per_block, **self.conv_kws)
            self.enc_blocks.append(c)
        
        prev_out_ch = filters[-1]
        skip_chs = filters[-2::-1]
        for skip_ch, out_ch in zip(skip_chs, dec_filters):
            in_ch = skip_ch + prev_out_ch if self.skip_connections else prev_out_ch
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_kws)
            prev_out_ch = out_ch
            self.dec_blocks.append(c)

    
    def forward(self):
        
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
                x = eqx.cat([x, conv_outputs[-i]], dim=1)
            x = conv_block(x)

        x = self.out_conv(x)

        if self.out_activation:
            x = self.out_fn(x)

        return x
    
        return None
