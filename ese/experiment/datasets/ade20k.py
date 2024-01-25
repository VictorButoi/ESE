# torch imports
import torch
# random imports
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class ADE20K(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.1
    preload: bool = False
    cities: Any = "all" 
    iters_per_epoch: Optional[int] = None
    transforms: Optional[Any] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        # Get the subjects from the splits
        samples = self._db["_splits"][self.split]
        sample_cities = self._db["_cities"]
        
        if self.cities != "all":
            assert isinstance(self.num_classes, list), "If not 'all', must specify the classes."
            self.samples = []
            self.sample_cities = []
            for (sample, class_id) in zip(samples, sample_cities):
                if class_id in self.cities:
                    self.samples.append(sample)
                    self.sample_cities.append(class_id)
        else:
            self.samples = samples 
            self.sample_cities = sample_cities 
        
        self.return_data_id = False
        # Control how many samples are in each epoch.
        self.num_samples = len(self.samples) if self.iters_per_epoch is None else self.iters_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        example_name = self.samples[key]
        # Get the class and associated label
        img, mask = self._db[example_name]
        # Get the class name
        if self.transforms:
            img, mask = self.transforms(img, mask)
        # Convert to float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        assert img.dtype == np.float32, "Img must be float32 (so augmenetation doesn't break)!"
        assert mask.dtype == np.float32, "Mask must be float32 (so augmentation doesn't break)!"

        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img),
            "label": torch.from_numpy(mask),
        }
        if self.return_data_id:
            return_dict["data_id"] = example_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"ADE20K/thunder_ade20k/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "ADE20K",
            "cities": self.cities,
            "split": self.split,
            "version": self.version
        }