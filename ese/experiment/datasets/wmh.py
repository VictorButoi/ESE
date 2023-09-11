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

    task: str 
    annotator: str = "observer_o12"
    axis: Literal[0, 1, 2] = 0
    split: Literal["train", "cal", "val", "test"] = "train"
    slicing: Literal["dense", "uniform", "midslice", "full"] = "dense"
    num_slices: int = 1
    version: float = 0.2
    preload: bool = False
    dataset: Literal["WMH"] = "WMH"
    slice_batch_size: Optional[int] = 1 
    transforms: Optional[List[Any]] = None

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
        allow_replacement = self.num_slices > len(label_amounts[label_amounts> 0])

        # Dense slice sampling means that you sample proportional to how much
        # label there is.
        if self.slicing == "dense":
            label_probs = label_amounts / np.sum(label_amounts)
            slice_indices = np.random.choice(np.arange(256), size=self.num_slices, p=label_probs, replace=allow_replacement)
        # Uniform slice sampling means that we sample all non-zero slices equally.
        elif self.slicing == "uniform":
            slice_indices = np.random.choice(np.where(label_amounts > 0)[0], size=self.num_slices, replace=allow_replacement)
        elif self.slicing == "full":
            slice_indices = np.arange(256)
        # Otherwise slice  down the middle.
        else:
            slice_indices = np.array([128])
        
        # Data object ensures first axis is the slice axis.
        img = img_vol[slice_indices, ...].astype(np.float32)
        mask = mask_vol[slice_indices, ...].astype(np.float32)

        assert img.dtype == np.float32, "Img must be float32!"
        assert mask.dtype == np.float32, "Mask must be float32!"

        if self.transforms:
            print("I go in here yes?")
            img, mask = self.transforms(img, mask)

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
