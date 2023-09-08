from dataclasses import dataclass
from typing import Any, Literal
import pathlib

# ionpy imports
from ionpy.util.validation import validate_arguments_init

# torch datasets
import torch
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection


@validate_arguments_init
@dataclass
class COCO(CocoDetection):

    split: Literal["train", "cal", "val"] = "train"
    root: pathlib.Path = pathlib.Path("/storage/vbutoi/datasets/COCO")
    transform: Any = None # For the image only
    transforms: Any = None # Takes in both image and mask
    target_transform: Any = None # Applied just to mask

    def __post_init__(self):
        if self.split == "train":
            path2data = self.root / "train2017"
            path2json = self.root / "annotations/instances_train2017.json"
        elif self.split in ["cal", "val"]:
            path2data = self.root / "val2017" 
            path2json = self.root / "annotations/instances_val2017.json"
        else:
            raise ValueError(f"Split {self.split} not recognized.")
        super(COCO, self).__init__(path2data, path2json, self.transform, self.target_transform, self.transforms)

    def __getitem__(self, key):
        # Load the original image and target (list of annotations)
        img, target = super(COCO, self).__getitem__(key)
        
        # Create an empty mask
        img = F.to_tensor(img) 
        mask = torch.zeros_like(img, dtype=torch.uint8)

        for ann in target:
            # COCO uses 'category_id' to indicate class of object
            # You might want to map category_id to consecutive integers starting from 1 if they aren't consecutive.
            category = ann['category_id']
            
            # Convert binary masks to PyTorch tensor
            m = F.to_tensor(self.coco.annToMask(ann)) * category
            mask = torch.where(m > 0, m, mask)

        # Apply transforms if given
        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask


    @property
    def signature(self):
        return {
            "dataset": "COCO",
            "split": self.split,
        }
