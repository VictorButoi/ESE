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
class OxfordPets(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"] = "train"
    version: float = 0.2
    preload: bool = False
    skip_classes: Optional[List[str]] = None
    transforms: Optional[List[Any]] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)

        # Get the subjects from the splits
        subjects = self._db["_splits"][self.split]
        classes = self._db["_classes"]

        if self.skip_classes:
            classes = [c for c in classes if c not in self.skip_classes]
            subjects = [subj for subj in subjects if subj.split("_")[:-1] in classes]

        self.classes = classes
        self.samples = subjects
        self.subjects = subjects


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        example_name = self.samples[key]
        img, mask = self._db[example_name]

        assert img.dtype == np.float32, "Img must be float32!"
        assert mask.dtype == np.int64, "Mask must be int64!" 

        if self.transforms:
            img, mask = self.transforms(img, mask)

        # Convert to torch tensors
        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())[None]

        return img, mask

    @property
    def _folder_name(self):
        return f"OxfordPets/thunder_oxfordpets/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "OxfordPets",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version
        }
