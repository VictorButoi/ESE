# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# ionpy imports
from ionpy.nn.nonlinearity import get_nonlinearity
# misc imports
import ast
from typing import Any, Literal, Optional, Tuple


class E2T(nn.Module):

    def __init__(
        self, 
        backbone_model
    ):
        super(E2T, self).__init__()
        # Determine the classes we need to use based on the dimensionality of the input.
        # The final fully connected layer.
        self.backbone = backbone_model

    def weights_init(self):
        pass

    def forward(self, image, **kwargs):
        _, x_feats = self.backbone.encode(image)
        print(x_feats.shape)
        raise ValueError("Not implemented yet.")

        # Return the output of the fc layer.
        return out 