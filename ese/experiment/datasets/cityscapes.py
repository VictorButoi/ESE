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
class CityScapes(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.1
    preload: bool = False
    cities: Any = "all" 
    iters_per_epoch: Optional[int] = None
    transforms: Optional[List[Any]] = None

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
        # Get the class conversion dictionary (From Calibration in Semantic Segmentation are we on the right) 
        class_conversion_dict = {
            7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
            23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18
            }
        self.label_map = torch.zeros(35, dtype=torch.int64) # 35 classes in total
        for old_label, new_label in class_conversion_dict.items():
            self.label_map[old_label] = new_label 
            
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

        # Data object ensures first axis is the slice axis.
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        # If we are remapping the labels, then we need to do that here.
        if self.label_map is not None:
            mask = self.label_map[mask]
        
        # Prepare the return dictionary.
        return_dict = {
            "img": img.float(),
            "label": mask.float(),
        }

        if self.return_data_id:
            return_dict["data_id"] = example_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"CityScapes/thunder_cityscapes/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "CityScapes",
            "cities": self.cities,
            "split": self.split,
            "version": self.version
        }