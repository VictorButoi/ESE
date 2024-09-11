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


def pad_to_resolution(arr, target_size):
    """
    Pads a numpy array to the given target size, which can be either a single number (same for all dimensions)
    or a tuple/list of integers corresponding to each dimension.

    Parameters:
    arr (numpy.ndarray): N-dimensional numpy array to be padded.
    target_size (int or tuple/list): Desired size for the padding. If a single integer is provided, 
                                     the array will be padded equally in all dimensions. If a tuple or 
                                     list is provided, it will pad each dimension accordingly.

    Returns:
    numpy.ndarray: Padded array.
    """
    # Get the current dimensions of the array
    current_size = arr.shape

    if len(current_size) == 0:
        raise ValueError("Input array must have at least one dimension.")

    # Handle the case where target_size is a single integer (same padding for all dimensions)
    if isinstance(target_size, int):
        target_size = [target_size] * len(current_size)
    elif isinstance(target_size, (tuple, list)):
        if len(target_size) != len(current_size):
            raise ValueError("Target size must have the same number of dimensions as the input array.")
    else:
        raise ValueError("Target size must be an integer or a tuple/list of integers.")

    # Assert that none of the dimensions are larger than the corresponding target sizes
    for i in range(len(current_size)):
        assert current_size[i] <= target_size[i], f"Dimension {i} of the array is larger than the target size."

    # Calculate the padding needed on each side for each dimension
    padding = []
    for i in range(len(current_size)):
        pad_before = (target_size[i] - current_size[i]) // 2
        pad_after = target_size[i] - current_size[i] - pad_before
        padding.append((pad_before, pad_after))

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

            
def get_max_slice_on_axis(img, seg, axis):
    all_axes = [0, 1, 2]
    # pop the axis from the list
    all_axes.pop(axis)
    # Get the maxslice of the seg_vol along the last axis
    label_per_slice = np.sum(seg, axis=tuple(all_axes))
    max_slice_idx = np.argmax(label_per_slice)
    # Get the image and segmentation as numpy arrays
    axis_max_img = np.take(img, max_slice_idx, axis=axis)
    axis_max_seg = np.take(seg, max_slice_idx, axis=axis)
    return axis_max_img, axis_max_seg 


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

        # keep track of the largest dimensions
        max_shape = [0, 0, 0]

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

            # Get the amount of segmentation in the image
            label_amount = np.count_nonzero(seg_vol_arr)
            if label_amount >= config.get('min_label_amount', 0):

                # If we have a pad then pad the image and segmentation
                if 'pad_to' in config:
                    # If 'pad_to' is a string then we need to convert it to a tuple
                    if isinstance(config['pad_to'], str):
                        config['pad_to'] = ast.literal_eval(config['pad_to'])
                    img_vol_arr = pad_to_resolution(img_vol_arr, config['pad_to'])
                    seg_vol_arr = pad_to_resolution(seg_vol_arr, config['pad_to'])

                if config.get('show_examples', False):
                    # Display the image and segmentation for each axis is a 2x3 grid.
                    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                    # Loop through the axes, plot the image and seg for each axis
                    for ax in range(3):
                        ax_max_img, ax_max_seg = get_max_slice_on_axis(img_vol_arr, seg_vol_arr, ax)
                        axs[0, ax].imshow(ax_max_img, cmap='gray')
                        axs[0, ax].set_title(f"Image on axis {ax}")
                        axs[1, ax].imshow(ax_max_seg, cmap='gray')
                        axs[1, ax].set_title(f"Segmentation on axis {ax}")
                    plt.show()

        #         # Get the max slice on the z axis
        #         if 'max_axis' in config:
        #             img, seg = get_max_slice_on_axis(img_vol_arr, seg_vol_arr, 2)
        #         else:
        #             img, seg = img_vol_arr, seg_vol_arr

        #         # Normalize the image to be between 0 and 1
        #         normalized_img = (img - img.min()) / (img.max() - img.min())
        #         # Get the proportion of the binary mask.
        #         gt_prop = np.count_nonzero(seg) / seg.size

        #         ## Save the datapoint to the database
        #         db[subj_name] = {
        #             "img": normalized_img, 
        #             "seg": seg,
        #             "gt_proportion": gt_prop
        #         } 
        #         subjects.append(subj_name)

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