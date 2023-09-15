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
        samples = self._db["_splits"][self.split]
        classes = self._db["_classes"]

        if self.skip_classes:
            classes = np.unique([c for c in classes if c not in self.skip_classes])
            samples = [subj for subj in samples if "_".join(subj.split("_")[:-1]) in classes]

        self.classes = classes
        self.class_map = {c: (i + 1) for i, c in enumerate(np.unique(classes))} # 1 -> 38 (0 background)
        self.samples = samples 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        example_name = self.samples[key]

        # Get the class and associated label
        img, mask = self._db[example_name]

        # Get the class name
        class_name = "_".join(example_name.split("_")[:-1])
        label = self.class_map[class_name]
        mask = (mask * label)[None]

        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        # Convert to float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        assert img.dtype == np.float32, "Img must be float32 (so augmenetation doesn't break)!"
        assert mask.dtype == np.float32, "Mask must be float32 (so augmentation doesn't break)!"

        # Convert to torch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

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
