# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init
# torch imports
import torch
# random imports
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@validate_arguments_init
@dataclass
class OASIS(ThunderDataset, DatapathMixin):

    axis: Literal[0, 1, 2]
    label_set: Literal["label4", "label35"]
    split: Literal["train", "cal", "val", "test"]
    slicing: str = "midslice"
    num_slices: int = 1
    replace: bool = False
    central_width: int = 32 
    version: float = 0.1
    binary: bool = False
    return_data_id: bool = False
    preload: bool = False
    slice_batch_size: Optional[int] = 1 
    iters_per_epoch: Optional[int] = None
    target_labels: Optional[List[int]] = None
    transforms: Optional[List[Any]] = None

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        self.subjects = self._db["_splits"][self.split]

        # If target labels is not None, then we need to remap the target labels to a contiguous set.
        if self.target_labels is not None:
            if self.label_set == "label4":
                self.label_map = torch.zeros(5, dtype=torch.int64)
            else:
                self.label_map = torch.zeros(36, dtype=torch.int64)
            for i, label in enumerate(self.target_labels):
                if self.binary:
                    self.label_map[label] = 1
                else:
                    self.label_map[label] = i
        else:
            assert not self.binary, "Binary labels require target labels to be specified."
            self.label_map = None
        
        # Control how many samples are in each epoch.
        self.num_samples = len(self.subjects) if self.iters_per_epoch is None else self.iters_per_epoch


    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.subjects)
        subj = self.subjects[key]
        subj_dict = self._db[subj]
        img_vol = subj_dict['image']
        mask_vol = subj_dict['mask']
        lab_amounts_per_slice = subj_dict['lab_amounts_per_slice']

        # Get the label_amounts
        label_amounts = np.zeros_like(lab_amounts_per_slice)
        lab_list = self.target_labels if self.target_labels is not None else lab_amounts_per_slice.keys()
        for label in lab_list:
            label_amounts += lab_amounts_per_slice[label]

        # Use this for slicing.
        vol_size = mask_vol.shape[0] # Typically 256
        # Slice the image and label volumes down the middle.
        if self.slicing == "midslice":
            slice_indices = np.array([128])
        # Sample the slice proportional to how much label they have.
        elif self.slicing == "dense":
            label_probs = self.label_amounts_per_slice[subj] / np.sum(self.label_amounts_per_slice[subj])
            slice_indices = np.random.choice(np.arange(vol_size), size=self.num_slices, p=label_probs, replace=self.replace)
        elif self.slicing == "uniform":
            slice_indices = np.random.choice(np.where(self.label_amounts_per_slice[subj] > 0)[0], size=self.num_slices, replace=self.replace)
        # Sample an image and label slice from around a central region.
        elif self.slicing == "central":
            central_slices = np.arange(128 - self.central_width, 128 + self.central_width)
            slice_indices = np.random.choice(central_slices, size=self.num_slices, replace=self.replace)
        elif self.slicing == "full_central":
            slice_indices = np.arange(128 - self.central_width, 128 + self.central_width)
        # Return the entire image and label volumes.
        elif self.slicing == "full":
            slice_indices = np.arange(256)
        # Throw an error if the slicing method is unknown.
        else:
            raise NotImplementedError(f"Unknown slicing method {self.slicing}")
        
        # Data object ensures first axis is the slice axis.
        img = img_vol[slice_indices, ...].astype(np.float32)
        mask = mask_vol[slice_indices, ...].astype(np.float32)

        if self.transforms:
            img, mask = self.transforms(img, mask)

        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img),
            "label": torch.from_numpy(mask),
        }

        # If we are remapping the labels, then we need to do that here.
        if self.label_map is not None:
            return_dict["label"] = self.label_map[return_dict["label"]]

        if self.return_data_id:
            return_dict["data_id"] = subj

        return return_dict

    @property
    def _folder_name(self):
        return f"OASIS/thunder_oasis/{self.version}/{self.axis}/{self.label_set}"

    @property
    def signature(self):
        return {
            "dataset": "OASIS",
            "split": self.split,
            "label_set": self.label_set,
            "axis": self.axis,
            "version": self.version,
        }
