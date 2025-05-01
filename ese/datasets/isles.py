# torch imports
import torch
# random imports
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.augmentation import init_monai_transforms
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class ISLES(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "cal_aug", "val", "test"]
    target: Literal['seg', 'temp', 'volume'] = 'seg' # Either optimize for segmentation or temperature.
    mode: Literal["rgb", "grayscale"] = "grayscale"
    version: float = 1.0 # 0.1 is maxslice, 1.0 is 3D
    preload: bool = False
    return_data_id: bool = False
    transforms: Optional[Any] = None
    num_examples: Optional[int] = None
    opt_temps_dir: Optional[str] = None
    examples: Optional[List[str]] = None
    aug_data_prob: Optional[float] = None # By default, we don't use augmented data.
    iters_per_epoch: Optional[Any] = None
    data_root: Optional[str] = None
    label_threshold: Optional[float] = None
    axis: Optional[Literal[0, 1, 2]] = None 
    min_fg_label: Optional[float] = 0.0 
    num_slices: Optional[int] = 1
    sample_slice_with_replace: Optional[bool] = False
    slicing: Optional[Literal["midslice", "maxslice"]] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # min_label_density
        subjects = self._db["_splits"][self.split]
        # Set these attributes to the class.
        self.samples = subjects
        self.subjects = subjects
        # Get the number of augmented examples, or set to 0 if not available.
        try:
            self.num_aug_examples = self._db["_num_aug_examples"][self.split]
        except:
            self.num_aug_examples = 0
        # Limit the number of examples available if necessary.
        assert not (self.num_examples and self.examples), "Only one of num_examples and examples can be set."

        if self.examples is not None:
            self.samples = [subj for subj in self.samples if subj in self.examples]
            self.subjects = self.samples

        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
            self.subjects = self.samples

        # Control how many samples are in each epoch.
        self.num_samples = len(self.subjects) if self.iters_per_epoch is None else self.iters_per_epoch
        print("Number of subjects: ", len(self.subjects))

        # Initialize the data transforms.
        # self.transforms_pipeline = init_album_transforms(self.transforms)
        self.transforms_pipeline = init_monai_transforms(self.transforms)

        # If opt temps dir is provided, then we need to load the optimal temperatures.
        if self.opt_temps_dir is not None:
            # Load the optimal temperatures from the json
            with open(self.opt_temps_dir, "r") as f:
                opt_temps_dict = json.load(f)
            self.opt_temps = {subj: torch.tensor([opt_temps_dict[subj]])for subj in self.subjects}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples) # Done for oversampling in the same epoch. This is the IDX of the sample.
        subject_name = self.subjects[key]

        # Get the image and mask
        example_obj = super().__getitem__(key)
        img, mask = example_obj["img"], example_obj["seg"]

        # Apply the label threshold
        if self.label_threshold is not None:
            mask = (mask > self.label_threshold).astype(np.float32)
        
        # If we are slicing then we need to do that here
        if self.slicing is not None:
            slice_indices = self.get_slice_indices(example_obj)
            img = np.take(img, slice_indices, axis=self.axis)
            mask = np.take(mask, slice_indices, axis=self.axis)

        # Apply the transforms, or a default conversion to tensor.
        # print("Before transform: ", img.shape, mask.shape)
        if self.transforms:
            transform_obj = self.transforms_pipeline(
                {"image": img, "mask": mask}
            )
            img, mask = transform_obj["image"], transform_obj["mask"]
        else:
            # We need to convert these image and masks to tensors at least.
            img = torch.tensor(img).unsqueeze(0)
            mask = torch.tensor(mask)
        # print("Post transform: ", img.shape, mask.shape)
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
            # We have a few options for what can be the target.
            if self.target == "temp":
                return_dict["label"] = self.opt_temps[subject_name]
            elif self.target == "volume":
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
        return f"ISLES/thunder_isles/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "ISLES",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version,
        }
