# Validating arguments imports
from dataclasses import dataclass
from IonPy.util import validate_arguments_init

# Torch imports
from torch.utils.data import Dataset

# Misc imports
import numpy as np
import os
import pathlib
import pickle
from PIL import Image
from tqdm import tqdm
from typing import Literal, List, Optional, Tuple, Union


# Dataset for the WMH Challenge dataset
@validate_arguments_init
@dataclass
class WMHDataset(Dataset):

    split: Literal["train", "val", "test"] = "train"
    reviewer: Literal["O1/O2", "O3", "O4"] = "O1/O2"
    datacenters: List[str] = ["Amsterdam", "Singapore", "Utrecht"]
    
    def __post_init__(self):
        # Call the constructor for PyTorch dataset
        super().__init__()
        self.root = pathlib.Path("/storage/vbutoi/datasets/WMH")
        filenames = []
        for dc in self.datacenters:
            # Amsterdam has a weird subfolder we will index into.
            if dc == "Amsterdam":
                dc += "GE3T"
            
            # Get the filenames
            filenames += list((self.root / dc / self.split / self.reviewer).glob("*.nii.gz"))
            




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None