# torch imports
import torch
# random imports

from dataclasses import dataclass
from typing import Any, List, Literal, Optional
import numpy as np

# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class WMH(ThunderDataset, DatapathMixin):

    axis: Literal[0, 1, 2]
    task: str 
    slicing: str
    split: Literal["train", "cal", "val", "test"]
    annotator: str = "observer_o12"
    num_slices: int = 1
    replace: bool = False
    central_width: int = 32 
    slice_batch_size: Optional[int] = 1 
    transforms: Optional[List[Any]] = None
    version: float = 0.2
    preload: bool = False

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)

        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        subj = self.subjects[key]
        subj_dict = self._db[subj]

        # Get the image and mask
        img_vol = subj_dict['image']
        mask_vol = subj_dict['masks'][self.annotator]

        label_amounts = subj_dict['pixel_proportions'][self.annotator]
        allow_replacement = self.replace or (self.num_slices > len(label_amounts[label_amounts> 0]))

        # Slice the image and label volumes down the middle.
        if self.slicing == "midslice":
            slice_indices = np.array([128])
        # Sample an image and label slice from around a central region.
        elif self.slicing == "central":
            central_slices = np.arange(128 - self.central_width, 128 + self.central_width)
            slice_indices = np.random.choice(central_slices, size=self.num_slices, replace=allow_replacement)
        # Sample the slice proportional to how much label they have.
        elif self.slicing == "dense":
            label_probs = label_amounts / np.sum(label_amounts)
            slice_indices = np.random.choice(np.arange(256), size=self.num_slices, p=label_probs, replace=allow_replacement)
        # Uniform slice sampling means that we sample all non-zero slices equally.
        elif self.slicing == "uniform":
            slice_indices = np.random.choice(np.where(label_amounts > 0)[0], size=self.num_slices, replace=allow_replacement)
        # Return the entire image and label volumes.
        elif self.slicing == "full":
            slice_indices = np.arange(256)
        # Throw an error if the slicing method is unknown.
        else:
            raise NotImplementedError(f"Unknown slicing method {self.slicing}")
        
        # Data object ensures first axis is the slice axis.
        img = img_vol[slice_indices, ...].astype(np.float32)
        mask = mask_vol[slice_indices, ...].astype(np.float32)

        assert img.dtype == np.float32, "Img must be float32 (so augmenetation doesn't break)!"
        assert mask.dtype == np.float32, "Mask must be float32 (so augmentation doesn't break)!"

        if self.transforms:
            img, mask = self.transforms(img, mask)

        # Convert to torch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask

    @property
    def _folder_name(self):
        return f"WMH/thunder_wmh/{self.version}/{self.task}/{self.axis}"

    @property
    def signature(self):
        return {
            "annotator": self.annotator,
            "dataset": "WMH",
            "resolution": self.resolution,
            "slicing": self.slicing,
            "split": self.split,
            "task": self.task,
            "version": self.version,
        }
