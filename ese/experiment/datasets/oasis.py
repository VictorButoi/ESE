# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init
# torch imports
import torch
# random imports
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@validate_arguments_init
@dataclass
class OASIS(ThunderDataset, DatapathMixin):

    axis: Literal[0, 1, 2]
    label_set: Literal["label4", "label35"]
    slicing: Literal["midslice", "central", "full"]
    split: Literal["train", "cal", "val", "test"]
    num_slices: int = 1
    replace: bool = False
    central_width: int = 32 
    version: float = 0.1
    preload: bool = False
    slice_batch_size: Optional[int] = 1 
    transforms: Optional[List[Any]] = None

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        subjects = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        subj = self.subjects[key]
        img_vol, mask_vol = self._db[subj]

        # Slice the image and label volumes down the middle.
        if self.slicing == "midslice":
            slice_indices = np.array([128])
        # Sample an image and label slice from around a central region.
        elif self.slicing == "central":
            central_slices = np.arange(128 - self.central_width, 128 + self.central_width)
            slice_indices = np.random.choice(central_slices, size=self.num_slices, replace=self.replace)
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

        return torch.from_numpy(img), torch.from_numpy(mask)

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
