# Misc imports
import sys
import numpy as np
from typing import Tuple
# Torch imports
import torch
from torch.utils.data import Dataset, IterableDataset
# Local imports
from .segment2d import Segment2D
# Ionpy imports
from ionpy.util import ShapeChecker


class RandomSupport(Dataset):
    def __init__(
        self, 
        dataset: Segment2D, 
        support_size: int, 
        replacement: bool = True,
        return_data_ids: bool = False
    ):
        self.dataset = dataset
        self.support_size = support_size
        self.replacement = replacement
        self.return_data_ids = return_data_ids

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:  # S, C, H, W
        rng = np.random.default_rng(seed)
        if self.replacement:
            idxs = rng.integers(len(self.dataset), size=self.support_size)
        else:
            idxs = rng.permutation(len(self.dataset))[: self.support_size]
        if self.return_data_ids:
            imgs, segs, data_ids = zip(*(self.dataset[i] for i in idxs))
            return torch.stack(imgs), torch.stack(segs), data_ids
        else:
            imgs, segs = zip(*(self.dataset[i] for i in idxs))
            return torch.stack(imgs), torch.stack(segs)
