# ionpy imports
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init
# torch imports
import torch
# random imports
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@validate_arguments_init
@dataclass
class OASIS(ThunderDataset):

    axis: Literal[0, 1, 2]
    label_set: Literal["label4", "label35"]
    split: Literal["train", "cal", "val", "test"]
    slicing: str = "midslice"
    num_slices: int = 1
    replace: bool = False
    central_width: int = 32 
    version: float = 0.1
    binary: bool = False
    preload: bool = False
    slice_batch_size: Optional[int] = 1 
    iters_per_epoch: Optional[int] = None
    target_labels: Optional[List[int]] = None
    transforms: Optional[List[Any]] = None

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        subjects = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects
        self.return_data_id = False
        # Control how many samples are in each epoch.
        self.num_samples = len(subjects) if self.iters_per_epoch is None else self.iters_per_epoch
        # If target labels is not None, then we need to remap the target labels to a contiguous set.
        if self.target_labels is not None:
            if self.binary: 
                self.label_map = {label: 1 for label in self.target_labels}
            else:
                self.label_map = {label: i for i, label in enumerate(self.target_labels)}
        else:
            assert not self.binary, "Binary labels require target labels to be specified."

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        subj = self.subjects[key]
        img_vol, mask_vol = self._db[subj]

       # if target labels are not None, then we need to remap the labels in the target labels
        # and everything else to background.
        if self.target_labels is not None:
            # Define a mapping function
            def remap_fn(value):
                # Map the value to the corresponding value in the dictionary, or 0 if not found
                return self.label_map.get(value, 0)
            # Use numpy.vectorize to apply the remap function to each element in the array
            remap_vectorized = np.vectorize(remap_fn)
            # Apply the remap function to the NumPy array
            mask_vol = remap_vectorized(mask_vol)

        label_amounts_per_slice = mask_vol.sum(axis=(1, 2))
        vol_size = mask_vol.shape[0] # Typically 256
        # Slice the image and label volumes down the middle.
        if self.slicing == "midslice":
            slice_indices = np.array([128])
        # Sample the slice proportional to how much label they have.
        elif self.slicing == "dense":
            label_probs = label_amounts_per_slice / np.sum(label_amounts_per_slice)
            slice_indices = np.random.choice(np.arange(vol_size), size=self.num_slices, p=label_probs, replace=self.replace)
        elif self.slicing == "uniform":
            slice_indices = np.random.choice(np.where(label_amounts_per_slice> 0)[0], size=self.num_slices, replace=self.replace)
        # Sample an image and label slice from around a central region.
        elif self.slicing == "central":
            central_slices = np.arange(128 - self.central_width, 128 + self.central_width)
            slice_indices = np.random.choice(central_slices, size=self.num_slices, replace=self.replace)
        elif self.slicing == "full_central":
            slice_indices = np.arange(128 - self.central_width, 128 + self.central_width)
        # Return the entire image and label volumes.
        elif self.slicing == "full":
            slice_indices = np.arange(256)
        # Throw an error if the slicing method is unknown.
        else:
            raise NotImplementedError(f"Unknown slicing method {self.slicing}")
        
        # Data object ensures first axis is the slice axis.
        img = img_vol[slice_indices, ...].astype(np.float32)
        mask = mask_vol[slice_indices, ...].astype(np.float32)

        if self.transforms:
            img, mask = self.transforms(img, mask)

        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img),
            "label": torch.from_numpy(mask),
        }
        if self.return_data_id:
            return_dict["data_id"] = subj

        return return_dict

    @property
    def _folder_name(self):
        return f"OASIS/thunder_oasis/{self.version}/{self.axis}/{self.label_set}"

    @property
    def signature(self):
        return {
            "dataset": "OASIS",
            "split": self.split,
            "label_set": self.label_set,
            "axis": self.axis,
            "version": self.version,
        }
