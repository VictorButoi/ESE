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


def resize_with_aspect_ratio(image, target_size=256):
    """
    Resize the image so that its shortest side is of the target size 
    while maintaining the aspect ratio.
    
    :param image: numpy array of shape (height, width, channels)
    :param target_size: desired size for the shortest side of the image
    :return: resized image
    """
    
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Calculate the scaling factor
    if height < width:
        scaling_factor = target_size / height
        new_height = target_size
        new_width = int(width * scaling_factor)
    else:
        scaling_factor = target_size / width
        new_width = target_size
        new_height = int(height * scaling_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Center-crop the longer side
    if resized_image.shape[:2] != (target_size, target_size):
        height, width = resized_image.shape[:2]
        y1 = (height - target_size) // 2
        y2 = y1 + target_size
        x1 = (width - target_size) // 2
        x2 = x1 + target_size
        cropped_image = resized_image[y1:y2, x1:x2]
    else:
        cropped_image = resized_image

    # Ensure that it's square
    assert cropped_image.shape[:2] == (target_size, target_size), f"Image shape is {cropped_image.shape}."
    
    return cropped_image 


def square_pad(img):
    if img.shape[0] != img.shape[1]:
        pad = abs(img.shape[0] - img.shape[1]) // 2
        if img.shape[0] > img.shape[1]:
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img


@validate_arguments
def data_splits(
    values: List[str], 
    splits: Tuple[float, float, float, float], 
    seed: int
) -> Tuple[List[str], List[str], List[str], List[str]]:

    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate entries found in values")

    # Super weird bug, removing for now, add up to 1!
    # if (s := sum(splits)) != 1.0:
    #     raise ValueError(f"Splits must add up to 1.0, got {splits}->{s}")

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


def thunderify_DRIVE(
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
        downsize_errors = []
        for example_name in tqdm(os.listdir(image_root), total=len(subj_list)):
            # Define the image_key
            key = "subject_" + example_name.split('_')[0]

            # Paths to the image and segmentation
            img_dir = proc_root / "images" / example_name 
            seg_dir = proc_root / "masks" / example_name.replace('_training.tif', '_manual1.gif')

            # Load the image and segmentation.
            img = np.array(Image.open(img_dir))
            seg = np.array(Image.open(seg_dir))
            seg = (seg - seg.min()) / (seg.max() - seg.min())

            # Square pad the img and seg to the larger dimension.
            img = square_pad(img)
            seg = square_pad(seg)

            # Get the ground-truth volumetric proportion.
            gt_proportion = np.count_nonzero(seg) / seg.size

            ########################
            # DOWNSIZING PROCEDURE.
            ########################

            # Do an absolutely minor amount of gaussian blurring to the seg ahead of time
            # so that the edges are a bit more fuzzy.
            seg = cv2.GaussianBlur(seg, (7, 7), 0)

            # Resize the image and segmentation to config["resize_to"]xconfig["resize_to"]
            img = resize_with_aspect_ratio(img, target_size=config["resize_to"])
            seg = resize_with_aspect_ratio(seg, target_size=config["resize_to"])

            # Convert to the right type
            img = img.astype(np.float32)
            seg = seg.astype(np.float32)

            # Move the channel axis to the front and normalize the labelmap to be between 0 and 1
            img = img.transpose(2, 0, 1)
            
            assert img.shape == (3, config["resize_to"], config["resize_to"]), f"Image shape isn't correct, got {img.shape}"
            assert seg.shape == (config["resize_to"], config["resize_to"]), f"Seg shape isn't correct, got {seg.shape}"
            assert np.count_nonzero(seg) > 0, "Label can't be empty."

            # Save the datapoint to the database
            db[key] = {
                "img": img, 
                "seg": seg,
                "gt_proportion": gt_proportion
            } 
            subjects.append(key)   

        subjects = sorted(subjects)
        splits = data_splits(subjects, splits_ratio, splits_seed)
        splits = dict(zip(("train", "cal", "val", "test"), splits))

        # Save the metadata
        db["_subjects"] = subjects
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="DRIVE",
            version=version,
            resolution=config["resize_to"],
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = splits
        db["_attrs"] = attrs

    