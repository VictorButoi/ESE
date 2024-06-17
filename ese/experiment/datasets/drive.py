# torch imports
import torch
# random imports
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
import numpy as np
import matplotlib.pyplot as plt
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class DRIVE(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.2
    preload: bool = False
    return_data_id: bool = False
    transforms: Optional[Any] = None
    iters_per_epoch: Optional[Any] = None
    label_threshold: Optional[float] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects
        # Control how many samples are in each epoch.
        self.num_samples = len(subjects) if self.iters_per_epoch is None else self.iters_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        subj_name = self.subjects[key]

        # Get the image and mask
        img, mask = super().__getitem__(key)

        # Add channel dimension to the mask
        mask = np.expand_dims(mask, axis=0)

        # Apply the label threshold
        if self.label_threshold is not None:
            mask = (mask > self.label_threshold).astype(np.float32)

        # Get the class name
        if self.transforms:
            transform_obj = self.transforms(image=img, mask=mask)
            img = transform_obj["image"]
            mask = transform_obj["mask"]
        
        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img).float(),
            "label": torch.from_numpy(mask).float(),
        }

        # Print the shapes
        if self.return_data_id:
            return_dict["data_id"] = subj_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"DRIVE/thunder_drive/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "DRIVE",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version,
        }
