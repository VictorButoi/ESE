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
    axis: Literal[0, 1, 2] = 0
    split: Literal["train", "cal", "val", "test"] = "train"
    slicing: Literal["dense", "uniform", "midslice"] = "dense"
    version: str = "v0.1"
    preload: bool = False
    samples_per_epoch: Optional[int] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()

        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects

    def __len__(self):
        if self.samples_per_epoch:
            return self.samples_per_epoch
        return len(self.samples)

    def __getitem__(self, key):
        img, seg = super().__getitem__(key)
        assert img.dtype == np.float32
        assert seg.dtype == np.float32

        img = img[None]

        if self.label is not None:
            seg = seg[self.label : self.label + 1]

        return torch.from_numpy(img), torch.from_numpy(seg)

    @property
    def _folder_name(self):
        return f"{self.dataset}/thunder_wmh/{self.task}"

    @classmethod
    def fromfile(cls, path, **kwargs):
        a = UniqueThunderReader(path)["_attrs"]
        task = f"{a['dataset']}/{a['group']}/{a['modality']}/{a['axis']}"
        return cls(
            task=task,
            resolution=a["resolution"],
            slicing=a["slicing"],
            version=a["version"],
            **kwargs,
        )

    @property
    def signature(self):
        return {
            "dataset": self.dataset,
            "task": self.task,
            "resolution": self.resolution,
            "split": self.split,
            "slicing": self.slicing,
            "version": self.version,
        }
