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
class WMH(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    hospital: Literal['Amsterdam', 'Singapore', 'Utrecht', 'Combined']
    annotator: str
    mode: Literal["rgb", "grayscale"] = "grayscale"
    target: Literal['seg', 'temp', 'volume'] = 'seg' # Either optimize for segmentation or temperature.
    version: float = 1.0
    preload: bool = False
    return_data_id: bool = False
    num_slices: Optional[int] = 1 
    transforms: Optional[Any] = None
    min_fg_label: Optional[int] = None
    num_examples: Optional[int] = None
    central_width: Optional[int] = None
    examples: Optional[List[str]] = None
    iters_per_epoch: Optional[Any] = None
    sample_slice_with_replace: bool = False
    label_threshold: Optional[float] = None
    data_root: Optional[str] = None
    axis: Optional[Literal[0, 1, 2]] = None 
    slicing: Optional[Literal["midslice", "maxslice", "central", "dense", "uniform", "dense_full", "full"]] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects

        # Limit the number of examples available if necessary.
        assert not (self.num_examples and self.examples), "Only one of num_examples and examples can be set."

        if self.examples is not None:
            self.samples = [subj for subj in self.samples if subj in self.examples]
            self.subjects = self.samples

        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
            self.subjects = self.samples
        print("Number of subjects: ", len(self.subjects))

        # Control how many samples are in each epoch.
        self.num_samples = len(self.subjects) if self.iters_per_epoch is None else self.iters_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        subject_name = self.subjects[key]

        # Get the image and mask
        subj_dict = super().__getitem__(key)
        img, mask = subj_dict['img'], subj_dict['seg']

        # Apply the label threshold
        if self.label_threshold is not None:
            mask = (mask > self.label_threshold).astype(np.float32)

        # Data object ensures first axis is the slice axis.
        if self.slicing is not None:
            slice_indices = self.get_slice_indices(subj_dict)
            img = np.take(img, slice_indices, axis=self.axis).squeeze(axis=self.axis)
            mask = np.take(mask, slice_indices, axis=self.axis).squeeze(axis=self.axis)

        # Get the class name
        if self.transforms:
            transform_obj = self.transforms(image=img, mask=mask)
            img, mask = transform_obj["image"], transform_obj["mask"]
        else:
            # We need to convert these image and masks to tensors at least.
            img = torch.tensor(img).unsqueeze(0)
            mask = torch.tensor(mask)
        # If the mode is rgb, then we need to duplicate the image 3 times.
        if self.mode == "rgb":
            img = torch.cat([img] * 3, axis=0)
        
        # Prepare the return dictionary.
        return_dict = {
            "img": img.float()
        }
        gt_seg = mask[None].float()

        # Determine which target we are optimizing for we want to always include
        # the ground truth segmentation, but sometimes as the prediction target
        # and sometimes as the label.
        if self.target == "seg":
            return_dict["label"] = gt_seg
        else:
            # If not using the segmentation as the target, we need to return the
            # segmentation as a different key.
            return_dict["gt_seg"] = gt_seg
            if self.target == "volume":
                raise NotImplementedError("Volume target not implemented.")
            elif self.target == "proportion":
                raise NotImplementedError("Volume target not implemented.")
            else:
                raise ValueError(f"Unknown target: {self.target}")

        # Optionally: We can add the data_id to the return dictionary.
        if self.return_data_id:
            return_dict["data_id"] = subject_name
        
        return return_dict
    
    def get_slice_indices(self, subj_dict):
        if 'pixel_proportions' in subj_dict:
            label_amounts = subj_dict['pixel_proportions'][self.annotator].copy()
        else:
            seg = subj_dict['seg']
            # Use the axis to find the sum amount of label along the other two axes.
            axes = [i for i in range(3) if i != self.axis]
            label_amounts = np.sum(seg, axis=tuple(axes))
        # Threshold if we can about having a minimum amount of label.
        if self.min_fg_label is not None and np.any(label_amounts > self.min_fg_label):
            label_amounts[label_amounts < self.min_fg_label] = 0

        allow_replacement = self.sample_slice_with_replace or (self.num_slices > len(label_amounts[label_amounts> 0]))
        vol_size = subj_dict['seg'].shape[self.axis] # Typically 245
        midvol_idx = vol_size // 2
        # Slice the image and label volumes down the middle.
        if self.slicing == "midslice":
            slice_indices = np.array([midvol_idx])
        elif self.slicing == "maxslice":
            # Get the slice with the most label.
            max_slice_idx = np.argmax(label_amounts)
            slice_indices = np.array([max_slice_idx])
        # Sample an image and label slice from around a central region.
        elif self.slicing == "central":
            central_slices = np.arange(midvol_idx - self.central_width, midvol_idx + self.central_width)
            slice_indices = np.random.choice(central_slices, size=self.num_slices, replace=allow_replacement)
        # Sample the slice proportional to how much label they have.
        elif self.slicing == "dense":
            label_probs = label_amounts / np.sum(label_amounts)
            slice_indices = np.random.choice(np.arange(vol_size), size=self.num_slices, p=label_probs, replace=allow_replacement)
        # Uniform slice sampling means that we sample all non-zero slices equally.
        elif self.slicing == "uniform":
            slice_indices = np.random.choice(np.where(label_amounts > 0)[0], size=self.num_slices, replace=allow_replacement)
        # Return the entire image and label volumes.
        elif self.slicing == "dense_full":
            slice_indices = np.where(label_amounts > 0)[0]
        elif self.slicing == "full":
            slice_indices = np.arange(vol_size)
        # Throw an error if the slicing method is unknown.
        else:
            raise NotImplementedError(f"Unknown slicing method {self.slicing}")
        #
        return slice_indices

    @property
    def _folder_name(self):
        return f"WMH/thunder_wmh/{self.version}/{self.hospital}/{self.annotator}"

    @property
    def signature(self):
        return {
            "dataset": "WMH",
            "annotator": self.annotator,
            "resolution": self.resolution,
            "slicing": self.slicing,
            "split": self.split,
            "hospital": self.hospital,
            "version": self.version,
        }
