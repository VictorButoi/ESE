import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional

import einops
import numpy as np
import parse
import torch
from parse import parse
from pydantic import validate_arguments

from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.thunder import UniqueThunderReader
from ionpy.util.validation import validate_arguments_init


def parse_task(task):
    return parse("{dataset}/{group}/{modality}/{axis}", task).named


@validate_arguments_init
@dataclass
class Segment2D(ThunderDataset, DatapathMixin):

    # task is (dataset, group, modality, axis)
    # - optionally label but see separate arg
    task: str
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

        # Signature to file checking
        file_attrs = self.attrs
        for key, val in parse_task(init_attrs["task"]).items():
            if file_attrs[key] != val:
                raise ValueError(
                    f"Attr {key} mismatch init:{val}, file:{file_attrs[key]}"
                )
        for key in ("resolution", "slicing", "version"):
            if init_attrs[key] != file_attrs[key]:
                raise ValueError(
                    f"Attr {key} mismatch init:{init_attrs[key]}, file:{file_attrs[key]}"
                )

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
        return f"megamedical/{self.version}/res{self.resolution}/{self.slicing}/{self.task}"

    @classmethod
    def frompath(cls, path, **kwargs):
        _, relpath = str(path).split("megamedical/")

        kwargs.update(
            parse("{version}/res{resolution:d}/{slicing:w}/{task}", relpath).named
        )
        return cls(**kwargs)

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
            "task": self.task,
            "resolution": self.resolution,
            "split": self.split,
            "slicing": self.slicing,
            "version": self.version,
            **parse_task(self.task),
        }
