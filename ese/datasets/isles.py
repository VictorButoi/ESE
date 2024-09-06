# torch imports
import torch
# random imports
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class ISLES(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.1
    preload: bool = False
    return_data_id: bool = False
    transforms: Optional[Any] = None
    return_gt_proportion: bool = False
    num_examples: Optional[int] = None
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
        # Limit the number of examples available if necessary.
        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
        # Control how many samples are in each epoch.
        self.num_samples = len(subjects) if self.iters_per_epoch is None else self.iters_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        subj_name = self.subjects[key]

        # Get the image and mask
        example_obj = super().__getitem__(key)
        img, mask = example_obj["img"], example_obj["seg"]

        # Apply the label threshold
        if self.label_threshold is not None:
            mask = (mask > self.label_threshold).astype(np.float32)

        # Get the class name
        if self.transforms:
            transform_obj = self.transforms(image=img, mask=mask)
            img = transform_obj["image"]
            mask = transform_obj["mask"]

        # Add channel dimension to the mask
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img).float(),
            "label": torch.from_numpy(mask).float(),
        }

        # Add some additional information.
        # Add some additional information.
        if self.return_gt_proportion:
            # NOTE: The key is mispelled in version 0.1
            return_dict["gt_proportion"] = example_obj["gt_propotion"]

        if self.return_data_id:
            return_dict["data_id"] = subj_name 
        
        return return_dict

    @property
    def _folder_name(self):
        return f"ISLES/thunder_isles/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "ISLES",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version,
        }
