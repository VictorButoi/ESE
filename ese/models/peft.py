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
        self.fc = nn.Linear()

    def forward(self, x):
        raise ValueError("Not implemented yet.")

        # Return the output of the fc layer.
        return out 