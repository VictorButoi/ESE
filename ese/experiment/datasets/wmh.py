from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch

# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class WMH(ThunderDataset, DatapathMixin):

    task: str 
    dataset: str
    annotator: str = "observer_o12"
    axis: Literal[0, 1, 2] = 0
    split: Literal["train", "cal", "val", "test"] = "train"
    slicing: Literal["dense", "uniform", "midslice"] = "dense"
    slice_batch_size: int = 1
    version: str = "v0.1"
    preload: bool = False
    samples_per_epoch: Optional[int] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)

        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects

    def __len__(self):
        if self.samples_per_epoch:
            return self.samples_per_epoch
        return len(self.samples)

    def __getitem__(self, key):
        subj = self.subjects[key]
        subj_dict = self._db[subj]

        # Get the image and mask
        img_vol = subj_dict['image']
        mask_vol = subj_dict['masks'][self.annotator]
        assert img_vol.dtype == np.float32
        assert mask_vol.dtype == np.float32

        # Dense slice sampling means that you sample proportional to how much
        # label there is.
        if self.slicing == "dense":
            all_axes = [0, 1, 2]
            all_axes.remove(self.axis)
            dist = np.sum(mask_vol, axis=tuple(all_axes))
            # If there are not enough non-zero slices, we allow replacement.
            allow_replacement = self.slice_batch_size > len(dist[dist > 0])
            # Same the slices proportional to the amount of label.
            slice_indices = np.random.choice(np.arange(256), size=self.slice_batch_size, p=dist/np.sum(dist), replace=allow_replacement)
            img_slice = np.take(img_vol, slice_indices, axis=self.axis)
            mask_slice = np.take(mask_vol, slice_indices, axis=self.axis)

        # Uniform slice sampling means that we sample all non-zero slices equally.
        elif self.slicing == "uniform":
            all_axes = [0, 1, 2]
            all_axes.remove(self.axis)
            dist = np.sum(mask_vol, axis=tuple(all_axes))
            chosen_slices = np.random.choice(np.where(dist > 0)[0], size=self.slice_batch_size, replace=False)
            img_slice = np.take(img_vol, chosen_slices, axis=self.axis)
            mask_slice = np.take(mask_vol, chosen_slices, axis=self.axis)

        # Otherwise slice both down the middle.
        else:
            img_slice = np.take(img_vol, 128, axis=self.axis)[None]
            mask_slice = np.take(mask_vol, 128, axis=self.axis)[None]

        def normalize_image(image):
            min_val = np.min(image)
            max_val = np.max(image)
            normalized_image = (image - min_val) / (max_val - min_val)
            return normalized_image

        # Make sure slice is between [0,1] and the correct dtype.
        img = normalize_image(img_slice)
        mask = mask_slice

        return torch.from_numpy(img), torch.from_numpy(mask)

    @property
    def _folder_name(self):
        return f"{self.dataset}/thunder_wmh/{self.task}"

    @property
    def signature(self):
        return {
            "dataset": self.dataset,
            "task": self.task,
            "resolution": self.resolution,
            "split": self.split,
            "slicing": self.slicing,
            "annotator": self.annotator,
            "version": self.version,
        }
