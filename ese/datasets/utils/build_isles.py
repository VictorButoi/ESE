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

from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments
from ...augmentation.gather import augmentations_from_config


def thunderify_ISLES(
    cfg: Config,
    splits: Optional[dict] = None
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

    # If we have augmentations in our config then we, need to make an aug pipeline
    if 'augmentations' in config:
        aug_pipeline = build_aug_pipeline(["augmentations"])

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

                # Get the max slice on the z axis
                if 'max_axis' in config:
                    img, seg = get_max_slice_on_axis(img_vol_arr, seg_vol_arr, 2)
                else:
                    img, seg = img_vol_arr, seg_vol_arr

                # Normalize the image to be between 0 and 1
                normalized_img = (img - img.min()) / (img.max() - img.min())
                # Get the proportion of the binary mask.
                gt_prop = np.count_nonzero(seg) / seg.size

                # TODO: If we are applying augmentation that effectively makes
                # synthetic data, then we need to do it here.
                raise NotImplementedError("Augmentation not implemented yet.")

                ## Save the datapoint to the database
                db[subj_name] = {
                    "img": normalized_img, 
                    "seg": seg,
                    "gt_proportion": gt_prop
                } 
                subjects.append(subj_name)

        subjects = sorted(subjects)
        # If splits aren't predefined then we need to create them.
        if splits is None:
            db_splits = data_splits(subjects, splits_ratio, splits_seed)
            db_splits = dict(zip(("train", "cal", "val", "test"), splits))
        else:
            db_splits = splits

        for split_key in db_splits:
            print(f"{split_key}: {len(db_splits[split_key])} samples")

        # Save the metadata
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="ISLES",
            version=version,
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = db_splits
        db["_attrs"] = attrs