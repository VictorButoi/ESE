from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np
import pathlib

# ionpy imports
from ionpy.util.validation import validate_arguments_init

# torch datasets
import torch
import torchvision.datasets as dset


@validate_arguments_init
@dataclass
class COCO:

    split: Literal["train", "cal", "val"] = "train"
    root: pathlib.Path = pathlib.Path("/storage/vbutoi/datasets/COCO")

    def __init__(self):

        if self.split == "train":
            path2data = self.root / "train2017"
            path2json = self.root / "annotations/panoptic_train2017.json"
        elif self.split in ["cal", "val"]:
            path2data = self.root / "val2017" 
            path2json = self.root / "annotations/panoptic_val2017.json"
        else:
            raise ValueError(f"Split {self.split} not recognized.")

        self.samples = dset.CocoDetection(
            root = path2data,
            annFile = path2json
        )
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

        return  
    @property
    def _folder_name(self):
        return f"COCO/thunder_wmh/{self.version}/{self.task}/{self.axis}"

    @property
    def signature(self):
        return {
            "dataset": "COCO",
            "resolution": self.resolution,
            "split": self.split,
        }
