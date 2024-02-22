# torch imports
import torch
# random imports
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class Shapes(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    subsplit: Literal[0, 1, 2, 3, 4] # These corresponds to different versions of the dataset for the same split.
    version: float = 0.1
    preload: bool = False
    labels: Optional[List[int]] = None
    return_data_id: bool = False
    iters_per_epoch: Optional[Any] = None
    transforms: Optional[Any] = None

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # Get the subjects from the splits
        samples = self._db["_splits"][self.split][self.subsplit]
        self.samples = samples 
        # Control how many samples are in each epoch.
        self.num_samples = len(self.samples) if self.iters_per_epoch is None else self.iters_per_epoch
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        example_name = self.samples[key]
        img, mask = super().__getitem__(key)

        # Zero out all labels that are not in the list.
        if self.labels is not None:
            mask = np.where(np.isin(mask, self.labels), mask, 0)

        # Apply the transforms to the numpy images.
        if self.transforms:
            img = img.transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image'].transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
            mask = transformed['mask'] # (H, W)

        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img),
            "label": torch.from_numpy(mask)[None], # Add a channel dimension 
        }

        if self.return_data_id:
            return_dict["data_id"] = example_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"Shapes/thunder_shapes/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "Shapes",
            "split": self.split,
            "subsplit": self.subsplit,
            "labels": self.labels,
            "version": self.version
        }