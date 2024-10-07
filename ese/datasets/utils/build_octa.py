import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from thunderpack import ThunderDB
from tqdm import tqdm
import cv2
from PIL import Image
from ionpy.util import Config

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def thunderify_OCTA(
    cfg: Config
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Append version to our paths
    version = str(config["version"])
    splits_seed = 42
    splits_ratio = (0.6, 0.2, 0.1, 0.1)

    # Append version to our paths
    proc_root = pathlib.Path(config["proc_root"])
    dst_dir = pathlib.Path(config["dst_dir"]) / version

    image_root = str(proc_root / 'images')

    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        subjects = [] 
        # Iterate through the examples.
        subj_list = list(os.listdir(image_root))
        for example_name in tqdm(os.listdir(image_root), total=len(subj_list)):
            # Define the image_key
            key = "subject_" + example_name.split('_')[0]

            # Paths to the image and segmentation
            img_dir = proc_root / "images" / example_name 
            seg_dir = proc_root / "masks" / example_name

            # Load the image and segmentation.
            raw_img = np.array(Image.open(img_dir))
            raw_seg = np.array(Image.open(seg_dir))

            # For this dataset, as it is non-binary, we have to do something more clever.
            # First we need to map our segmentation to a binary segmentation.
            unique_labels = np.unique(raw_seg)
            mask_dict = {} 
            gt_prop_dict = {}

            # If the unique labels are greater than 2, then we have to do some work.
            square_img = square_pad(raw_img)
            square_raw_seg = square_pad(raw_seg)

            # Remove the axes and ticks
            if square_img.shape[0] != config["resize_to"]:
                square_img = resize_with_aspect_ratio(square_img, target_size=config["resize_to"]).astype(np.float32)

            for lab in unique_labels:
                # If the lab != 0, then we make a binary mask.
                if lab != 0:
                    # Make a binary mask.
                    binary_mask = (square_raw_seg == lab).astype(np.float32)

                    # Get the proportion of the binary mask.
                    gt_prop = np.count_nonzero(binary_mask) / binary_mask.size
                    gt_prop_dict[lab] = gt_prop

                    # If we are resizing then we need to smooth the mask.
                    if square_raw_seg.shape[0] != config["resize_to"]:
                        smooth_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
                        # Now we process the mask in our standard way.
                        proc_mask = resize_with_aspect_ratio(smooth_mask, target_size=config["resize_to"])
                    else:
                        proc_mask = binary_mask

                    # Renormalize the mask to be between 0 and 1.
                    norm_mask = (proc_mask - proc_mask.min()) / (proc_mask.max() - proc_mask.min())

                    # Store the mask in the mask_dict.
                    mask_dict[lab] = norm_mask.astype(np.float32)
                
            assert square_img.shape == (config["resize_to"], config["resize_to"]), f"Image shape isn't correct, got {square_img.shape}"

            # Save the datapoint to the database
            db[key] = {
                "img": square_img, 
                "seg": mask_dict,
                "gt_proportion": gt_prop_dict 
            } 
            subjects.append(key)

        subjects = sorted(subjects)
        splits = data_splits(subjects, splits_ratio, splits_seed)
        splits = dict(zip(("train", "cal", "val", "test"), splits))
        for split_key in splits:
            print(f"{split_key}: {len(splits[split_key])} samples")

        # Save the metadata
        db["_subjects"] = subjects
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="OCTA_6M",
            version=version,
            resolution=config["resize_to"],
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = splits
        db["_attrs"] = attrs

        