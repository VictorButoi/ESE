import os
import ast
import cv2
import time
import pathlib
import voxel as vx
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from ionpy.util import Config
from thunderpack import ThunderDB

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def pad_to_square_resolution(arr, target_size):
    """
    Pads a 2D numpy array to the given square target_size.

    Parameters:
    arr (numpy.ndarray): 2D numpy array to be padded.
    target_size (int): Desired size for the square padding.

    Returns:
    numpy.ndarray: Padded 2D array.
    """
    # Get the current dimensions of the array
    current_size = arr.shape

    if len(current_size) != 2:
        raise ValueError("Input array must be 2D.")

    # Assert that neither of the dimensions are larger than the target size
    assert current_size[0] <= target_size, "Height of the image is larger than the target size."
    assert current_size[1] <= target_size, "Width of the image is larger than the target size."
    
    # Check if target size is large enough
    if target_size < max(current_size):
        raise ValueError("Target size must be greater than or equal to the current array size.")
    
    # Calculate the padding needed on each side
    pad_y = (target_size - current_size[0]) // 2
    pad_x = (target_size - current_size[1]) // 2
    
    # Create padding dimensions: ((top, bottom), (left, right))
    padding = ((pad_y, target_size - current_size[0] - pad_y), 
               (pad_x, target_size - current_size[1] - pad_x))
    
    # Pad the array with zeros
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=0)
    
    return padded_arr

@validate_arguments
def data_splits(
    values: List[str], 
    splits: Tuple[float, float, float, float], 
    seed: int
) -> Tuple[List[str], List[str], List[str], List[str]]:

    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate entries found in values")

    train_size, cal_size, val_size, test_size = splits
    values = sorted(values)
    # First get the size of the test splut
    traincalval, test = train_test_split(values, test_size=test_size, random_state=seed)
    # Next size of the val split
    val_ratio = val_size / (train_size + cal_size + val_size)
    traincal, val = train_test_split(traincalval, test_size=val_ratio, random_state=seed)
    # Next size of the cal split
    cal_ratio = cal_size / (train_size + cal_size)
    train, cal = train_test_split(traincal, test_size=cal_ratio, random_state=seed)

    assert sorted(train + cal + val + test) == values, "Missing Values"

    return (train, cal, val, test)


def thunderify_ISLES(
    cfg: Config
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Append version to our paths
    version = str(config["version"])
    splits_seed = 42
    splits_ratio = (0.6, 0.2, 0.1, 0.1)

    # Append version to our paths
    proc_root = pathlib.Path(config["root"]) / 'raw_data' / 'ISLES_22'
    dst_dir = pathlib.Path(config["root"]) / config["dst_folder"] / version

    isl_img_root = proc_root / 'cropped_images'
    isl_seg_root = proc_root / 'unzipped_archive' / 'derivatives'

    ## Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
                
        # Iterate through the examples.
        # Key track of the ids
        subjects = [] 
        subj_list = list(os.listdir(isl_img_root))
        for subj_name in tqdm(subj_list, total=len(subj_list)):

            # Paths to the image and segmentation
            img_dir = isl_img_root / subj_name / 'ses-0001' / 'dwi' / f'{subj_name}_ses-0001_dwi_cropped.nii.gz' 
            seg_dir = isl_seg_root / subj_name / 'ses-0001' / f'{subj_name}_ses-0001_msk.nii.gz' 

            # Load the volumes using voxel
            img_vol = vx.load_volume(img_dir)
            # Load the seg and process to match the image
            raw_seg_vol = vx.load_volume(seg_dir)
            seg_vol = raw_seg_vol.resample_like(img_vol, mode='nearest')

            # Get the tensors from the vol objects
            img_vol_arr = img_vol.tensor.numpy().squeeze()
            seg_vol_arr = seg_vol.tensor.numpy().squeeze()
            
            def get_max_slice_on_axis(img, seg, axis):
                all_axes = [0, 1, 2]
                # pop the axis from the list
                all_axes.pop(axis)
                # Get the maxslice of the seg_vol along the last axis
                label_per_slice = np.sum(seg, axis=tuple(all_axes))
                max_slice_idx = np.argmax(label_per_slice)
                # Get the image and segmentation as numpy arrays
                max_img = np.take(img, max_slice_idx, axis=axis)
                max_seg = np.take(seg, max_slice_idx, axis=axis)
                return max_img, max_seg 

            # Display the image and segmentation for each axis is a 2x3 grid.
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            # Loop through the axes, plot the image and seg for each axis
            for ax in range(3):
                max_img, max_seg = get_max_slice_on_axis(img_vol_arr, seg_vol_arr, ax)
                axs[0, ax].imshow(max_img, cmap='gray')
                axs[0, ax].set_title(f"Image on axis {ax}")
                axs[1, ax].imshow(max_seg, cmap='gray')
                axs[1, ax].set_title(f"Segmentation on axis {ax}")
            plt.show()


        #     # Normalize the image to be between 0 and 1
        #     max_img = (max_img - max_img.min()) / (max_img.max() - max_img.min())

        #     # If we have a pad then pad the image and segmentation
        #     if 'pad_to' in config:
        #         pad_size = config['pad_to']
        #         max_img = pad_to_square_resolution(max_img, pad_size)
        #         max_seg = pad_to_square_resolution(max_seg, pad_size)
            
        #     # Get the proportion of the binary mask.
        #     gt_prop = np.count_nonzero(max_seg) / max_seg.size

        #     ## Save the datapoint to the database
        #     db[subj_name] = {
        #         "img": max_img, 
        #         "seg": max_seg,
        #         "gt_propotion": gt_prop
        #     } 
        #     subjects.append(subj_name)

        # subjects = sorted(subjects)
        # splits = data_splits(subjects, splits_ratio, splits_seed)
        # splits = dict(zip(("train", "cal", "val", "test"), splits))

        # for split_key in splits:
        #     print(f"{split_key}: {len(splits[split_key])} samples")

        # # Save the metadata
        # db["_subjects"] = subjects
        # db["_splits"] = splits
        # db["_splits_kwarg"] = {
        #     "ratio": splits_ratio, 
        #     "seed": splits_seed
        #     }
        # attrs = dict(
        #     dataset="ISLES",
        #     version=version,
        # )
        # db["_subjects"] = subjects
        # db["_samples"] = subjects
        # db["_splits"] = splits
        # db["_attrs"] = attrs