from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
from parse import parse

# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.thunder import UniqueThunderReader
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class Segment2D(ThunderDataset, DatapathMixin):

    task: str 
    dataset: str
    annotator: str = "observer_o12"
    axis: Literal[0, 1, 2] = 0
    split: Literal["train", "cal", "val", "test"] = "train"
    slicing: Literal["dense", "uniform", "midslice"] = "dense"
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
        assert img.dtype == np.float32
        assert seg.dtype == np.float32
        print(img_vol.shape)
        print(mask_vol.shape)

        # Dense slice sampling means that you sample proportional to how much
        # label there is.
        if self.slicing == "dense":
            all_axes = [0, 1, 2]
            all_axes.remove(self.axis)
            dist = np.sum(mask_vol, axis=tuple(all_axes))
            slice_idx = np.random.choice(np.arange(256), p=dist/np.sum(dist))
            img_slice = np.take(img_vol, slice_idx, axis=self.axis)
            mask_slice = np.take(mask_vol, slice_idx, axis=self.axis)
        # Uniform slice sampling means that we sample all non-zero slices equally.
        elif self.slicing == "uniform":
            all_axes = [0, 1, 2]
            all_axes.remove(self.axis)
            dist = np.sum(mask_vol, axis=tuple(all_axes))
            chosen_slice = np.random.choice(np.where(dist > 0)[0])
            img_slice = np.take(img_vol, chosen_slice, axis=self.axis)
            mask_slice = np.take(mask_vol, chosen_slice, axis=self.axis)
        # Otherwise slice both down the middle.
        else:
            img_slice = np.take(img_vol, 128, axis=self.axis)
            mask_slice = np.take(mask_vol, 128, axis=self.axis)

        def normalize_image(image):
            min_val = np.min(image)
            max_val = np.max(image)
            normalized_image = (image - min_val) / (max_val - min_val)
            return normalized_image

        # Make sure slice is between [0,1] and the correct dtype.
        img = normalize_image(img_slice)[None]
        mask = mask_slice[None]

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
