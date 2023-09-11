# misc imports
import numpy as np
from dataclasses import dataclass
from typing import Any, Literal
import pathlib
import json

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
    transforms: Any = None # Takes in both image and mask

    def __post_init__(self):
        if self.split == "train":
            path2data = self.root / "train2017"
            path2json = self.root / "annotations/instances_train2017.json"
        elif self.split in ["cal", "val"]:
            path2data = self.root / "val2017" 
            path2json = self.root / "annotations/instances_val2017.json"
        else:
            raise ValueError(f"Split {self.split} not recognized.")
        super(COCO, self).__init__(path2data, path2json)

        # Check if the cache file exists
        cache_file = self.root / "label_info.json"
        # Check if the cache file exists
        try:
            with open(cache_file, 'r') as f:
                self.id_to_newid = json.load(f)
        except FileNotFoundError:
            # If not, create the mapping and save to cache file
            category_ids = set()
            for img_id in self.coco.getImgIds():
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                for ann in anns:
                    category_ids.add(ann['category_id'])

            category_ids = sorted(list(category_ids))
            self.id_to_newid = {str(id): i for i, id in enumerate(category_ids)}

            # Save the mapping to a cache file
            with open(cache_file, 'w') as f:
                json.dump(self.id_to_newid, f)

    def __getitem__(self, key):
        # Load the original image and target (list of annotations)
        img, target = super(COCO, self).__getitem__(key)

        # Create an empty mask
        img = F.to_tensor(img) 
        mask = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64)[None]

        for ann in target:
            # COCO uses 'category_id' to indicate class of object
            # You might want to map category_id to consecutive integers starting from 1 if they aren't consecutive.
            category = self.id_to_newid[str(ann['category_id'])]
            
            # Convert binary masks to PyTorch tensor
            mask_area = F.to_tensor(self.coco.annToMask(ann)).bool()
            mask[mask_area] = category

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
