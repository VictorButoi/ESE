# Misc imports
import json
import os
import ast
import time
import torch
import pathlib
import voxel as vx
import numpy as np
from PIL import Image
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from thunderpack import ThunderDB
from typing import List, Tuple, Optional
import imageio.v2 as imageio  # newer versions of imageio

# Ionpy imports
from ionpy.util import Config
# Local imports
from ...augmentation.pipeline import build_aug_pipeline
from .utils_for_build import (
    data_splits,
    vis_3D_subject,
    normalize_image,
    pairwise_aug_npy,
    pad_to_resolution
)

def resize_volume(volume, target_shape):
    zoom_factors = [
        target_shape[0] / volume.shape[0],  # height
        target_shape[1] / volume.shape[1],  # width
    ]
    if len(target_shape) == 3:
        zoom_factors.append(1.0) # depth (keep fixed)
    resized = zoom(volume, zoom=zoom_factors, order=1)  # order=1 = trilinear interpolation
    return resized

def thunderify_ISLES(
    cfg: Config,
    splits: Optional[dict] = {},
    splits_kwarg: Optional[dict] = None
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Append version to our paths
    version = str(config["version"])
    # Append version to our paths
    proc_root = pathlib.Path(config["root"]) / 'raw_data' / 'ISLES_22'
    dst_dir = pathlib.Path(config["root"]) / config["dst_folder"] / version

    isl_img_root = proc_root / 'cropped_images'
    isl_seg_root = proc_root / 'unzipped_archive' / 'derivatives'

    # If we have augmentations in our config then we, need to make an aug pipeline
    if 'augmentations' in config:
        aug_pipeline = build_aug_pipeline(config["augmentations"])
    else:
        aug_pipeline = None

    ## Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
                
        # Iterate through the examples.
        # Key track of the ids
        subjects = [] 
        aug_split_samples = []
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
                
                # If we want to resize the image and segmentation, then we do that
                # here.
                if 'resize_to' in config:
                    # If 'resize_to' is a string then we need to convert it to a tuple
                    if isinstance(config['resize_to'], str):
                        config['resize_to'] = ast.literal_eval(config['resize_to'])
                    img_vol_arr = resize_volume(img_vol_arr, config['resize_to'])
                    seg_vol_arr = resize_volume(seg_vol_arr, config['resize_to'])

                # Normalize the image to be between 0 and 1
                normalized_img_arr = normalize_image(img_vol_arr)

                # Get the proportion of the binary mask.
                gt_prop = np.count_nonzero(seg_vol_arr) / seg_vol_arr.size

                if config.get('show_examples', False):
                    vis_3D_subject(normalized_img_arr, seg_vol_arr)
                
                print("img size ", img_vol_arr.shape)
                print("seg size ", seg_vol_arr.shape)

                # We actually can have a distinction between samples and subjects!!
                # Splits are done at the subject level, so we need to keep track of the subjects.
                db[subj_name] = {
                    "img": normalized_img_arr, 
                    "seg": seg_vol_arr,
                    "gt_proportion": gt_prop
                } 
                subjects.append(subj_name)

                #####################################################################################
                # AUGMENTATION SECTION: USED FOR ADDING ADDITIONAL AUGMENTED SAMPLES TO THE DATASET #
                #####################################################################################
                # If we are applying augmentation that effectively makes
                # synthetic data, then we need to do it here.
                if aug_pipeline is not None and subj_name in splits.get("cal", []):
                    aug_split_samples.append(subj_name) # Add the original subject to the augmented split
                    for aug_idx in range(config["aug_examples_per_subject"]):
                        augmented_img_arr, augmented_seg_arr = pairwise_aug_npy(normalized_img_arr, seg_vol_arr, aug_pipeline)

                        # Calculate the new proportion of the binary mask.
                        aug_gt_prop = np.count_nonzero(augmented_seg_arr) / augmented_seg_arr.size

                        if config.get('show_examples', False):
                            vis_3D_subject(augmented_img_arr, augmented_seg_arr)

                        # Modify the name of the subject to reflect that it is an augmented sample
                        aug_subj_name = f"{subj_name}_aug_{aug_idx}"
                        # We actually can have a distinction between samples and subjects!!
                        # Splits are done at the subject level, so we need to keep track of the subjects.
                        db[aug_subj_name] = {
                            "img": augmented_img_arr, 
                            "seg": augmented_seg_arr,
                            "gt_proportion": aug_gt_prop
                        } 
                        aug_split_samples.append(aug_subj_name)

        subjects = sorted(subjects)
        # If splits aren't predefined then we need to create them.
        if splits == {}:
            splits_seed = 42
            splits_ratio = (0.6, 0.2, 0.1, 0.1)
            db_splits = data_splits(subjects, splits_ratio, splits_seed)
            db_splits = dict(zip(("train", "cal", "val", "test"), db_splits))
        else:
            splits_seed = splits_kwarg["seed"]
            splits_ratio = splits_kwarg["ratio"]
            db_splits = splits
        
        # If aug_split_samples is not empty then we add to as its own split
        if len(aug_split_samples) > 0:
            db_splits["cal_aug"] = aug_split_samples

        # Print the number of samples in each split for debugging purposes.
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
        db["_num_aug_examples"] = {
            "train": 0,
            "cal": config.get("aug_examples_per_subject", 0),
            "val": 0,
            "test": 0
        }
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = db_splits
        db["_attrs"] = attrs


def process_and_save(
    img_path,
    seg_path,
    img_out_path,
    seg_out_path,
    pad_to,
    resize_to,
    min_label_amount=0,
    show_examples=False,
):
    # --- Load ---
    # Load the volumes using voxel
    img = vx.load_volume(img_path)
    # Load the seg and process to match the image
    raw_seg_vol = vx.load_volume(seg_path)
    seg = raw_seg_vol.resample_like(img, mode='nearest')
    # Get the properties that the image has

    img_data = img.tensor.numpy().squeeze()
    seg_data = seg.tensor.numpy().squeeze()

    label_amount = np.count_nonzero(seg_data)
    if label_amount >= min_label_amount:

        # Slice the volume by the max slice on the last axis.
        lab_per_slice = np.sum(seg_data, axis=(0, 1))
        max_slice_idx = np.argmax(lab_per_slice)

        img_slice = img_data[:, :, max_slice_idx]
        seg_slice = seg_data[:, :, max_slice_idx]

        img_slice = pad_to_resolution(img_slice, pad_to)
        seg_slice = pad_to_resolution(seg_slice, pad_to)

        # --- Resize ---
        img_slice = resize_volume(img_slice, resize_to)
        seg_slice = resize_volume(seg_slice, resize_to)

        # --- Normalize ---
        img_slice = (normalize_image(img_slice) * 255).astype(np.uint8)
        # --- Binarize the segmentation ---
        seg_slice = np.where(seg_slice > 0, 1, 0).astype(np.uint8)

        if show_examples:
            f, axarr = plt.subplots(1, 2, figsize=(10, 5))
            im = axarr[0].imshow(img_slice, cmap='gray')
            axarr[0].set_title("Image")
            f.colorbar(im, ax=axarr[0])
            se =axarr[1].imshow(seg_slice, cmap='gray')
            axarr[1].set_title("Segmentation")
            f.colorbar(se, ax=axarr[1])
            plt.show()

        # Write out the image and segmentation
        imageio.imwrite(img_out_path, img_slice)
        imageio.imwrite(seg_out_path, seg_slice)


def nnunetize_ISLES_resave(
    cfg,
    splits: Optional[dict] = {},
):
    config = cfg.to_dict()

    proc_root = pathlib.Path(config["root"]) / 'raw_data' / 'ISLES_22'
    dst_dir = pathlib.Path(config["nnunet_dst_folder"])

    isl_img_root = proc_root / 'cropped_images'
    isl_seg_root = proc_root / 'unzipped_archive' / 'derivatives'

    out_tr_img_root = dst_dir / 'imagesTr'
    out_tr_seg_root = dst_dir / 'labelsTr'

    out_ts_img_root = dst_dir / 'imagesTs'
    out_ts_seg_root = dst_dir / 'labelsTs'

    # Prepare output folders
    out_tr_img_root.mkdir(parents=True, exist_ok=True)
    out_tr_seg_root.mkdir(parents=True, exist_ok=True)

    out_ts_img_root.mkdir(parents=True, exist_ok=True)
    out_ts_seg_root.mkdir(parents=True, exist_ok=True)

    subj_list = list(os.listdir(isl_img_root))

    # Print the total number of training subjects by the len
    # of the subjects in the splits
    for split_key in splits:
        print(f"{split_key}: {len(splits[split_key])} samples")

    for subj_name in tqdm(subj_list, total=len(subj_list)):

        img_path = isl_img_root / subj_name / 'ses-0001' / 'dwi' / f'{subj_name}_ses-0001_dwi_cropped.nii.gz'
        seg_path = isl_seg_root / subj_name / 'ses-0001' / f'{subj_name}_ses-0001_msk.nii.gz'

        if subj_name not in splits.get("test"):
            out_img_root = out_tr_img_root
            out_seg_root = out_tr_seg_root
        else:
            out_img_root = out_ts_img_root
            out_seg_root = out_ts_seg_root

        img_out_path = out_img_root / f'{subj_name}_0000.png'
        seg_out_path = out_seg_root / f'{subj_name}.png'

        pad_to = ast.literal_eval(config['pad_to']) if isinstance(config['pad_to'], str) else config['pad_to']
        resize_to = ast.literal_eval(config['resize_to']) if isinstance(config['resize_to'], str) else config['resize_to']

        process_and_save(
            img_path, 
            seg_path, 
            img_out_path, 
            seg_out_path,
            pad_to, 
            resize_to, 
            min_label_amount=config.get('min_label_amount', 0),
            show_examples=config.get('show_examples', False)
        )
    
    # Write this out to a json file.
    json_dict = {
        "channel_names": {  # formerly modalities
            "0": "DWI", 
        }, 
        "labels": {  # THIS IS DIFFERENT NOW!
            "background": 0,
            "foreground": 1,
        }, 
        "numTraining": 222,
        "file_ending": ".png"
    }
    json_path = dst_dir / 'dataset.json'
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)

    print(f"Saved dataset.json to {json_path}")


def thunderify_ISLES_frompng(
    cfg: Config,
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Append version to our paths
    version = str(config["version"])

    # Append version to our paths
    proc_root = pathlib.Path(config["root"])
    dst_dir = pathlib.Path(config["dst_folder"]) / version

    # Load the splits file
    split_file = cfg['splits_root'] + "/splits_final.json"
    with open(split_file, 'r') as f:
        db_splits = json.load(f)[0]
    # We need to load the names of all the test files for the final split
    test_dir = proc_root / 'imagesTs'
    test_list = [image_name.split("_")[0] for image_name in os.listdir(test_dir)]
    db_splits["test"] = test_list

    subjects = [] 
    # Get the subject list by combining the train and test splits
    subj_list = []
    for split_key in db_splits:
        subj_list += db_splits[split_key]

    ## Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:

        for subj_name in tqdm(subj_list, total=len(subj_list)):

            # Paths to the image and segmentation
            if subj_name in db_splits["test"]:
                img_dir = proc_root / 'imagesTs' / (subj_name + "_0000.png")
                seg_dir = proc_root / 'labelsTs' / (subj_name + ".png")
            else:
                img_dir = proc_root / 'imagesTr' / (subj_name + "_0000.png")
                seg_dir = proc_root / 'labelsTr' / (subj_name + ".png")

            # Get the tensors from the vol objects
            img_vol_arr = np.array(Image.open(img_dir))
            seg_vol_arr = np.array(Image.open(seg_dir))
            # Get the amount of segmentation in the image
            label_amount = np.count_nonzero(seg_vol_arr)
            if label_amount >= config.get('min_label_amount', 0):

                if 'resize_to' in config:
                    # If 'resize_to' is a string then we need to convert it to a tuple
                    if isinstance(config['resize_to'], str):
                        config['resize_to'] = ast.literal_eval(config['resize_to'])
                    img_vol_arr = resize_volume(img_vol_arr, config['resize_to'])
                    seg_vol_arr = resize_volume(seg_vol_arr, config['resize_to'])

                # Normalize the image to be between 0 and 1
                img_vol_arr = normalize_image(img_vol_arr)

                # We can visaulize these as an option.
                if config.get('show_examples', False):
                    f, axarr = plt.subplots(1, 2, figsize=(10, 5))
                    im = axarr[0].imshow(img_vol_arr, cmap='gray')
                    axarr[0].set_title("Image")
                    f.colorbar(im, ax=axarr[0])
                    se =axarr[1].imshow(seg_vol_arr, cmap='gray')
                    axarr[1].set_title("Segmentation")
                    f.colorbar(se, ax=axarr[1])
                    plt.show()

                # Get the proportion of the binary mask.
                gt_prop = np.count_nonzero(seg_vol_arr) / seg_vol_arr.size

                # We actually can have a distinction between samples and subjects!!
                # Splits are done at the subject level, so we need to keep track of the subjects.
                db[subj_name] = {
                    "img": img_vol_arr, 
                    "seg": seg_vol_arr,
                    "gt_proportion": gt_prop
                } 
                subjects.append(subj_name)

        subjects = sorted(subjects)
        
        # Print the number of samples in each split for debugging purposes.
        for split_key in db_splits:
            print(f"{split_key}: {len(db_splits[split_key])} samples")

        # Save the metadata
        attrs = dict(
            dataset="ISLES",
            version=version,
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = db_splits
        db["_attrs"] = attrs

