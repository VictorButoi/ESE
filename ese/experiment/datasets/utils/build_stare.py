import os
import io
import gzip
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


def shrink_boundary(binary_mask, pixels=15):
    """
    Removes pixels from the boundary of objects in a binary mask.

    Parameters:
    - binary_mask (np.array): A binary image where the object is represented by 255 and the background is 0.
    - pixels (int): The number of pixels to remove from the boundary.

    Returns:
    - np.array: A new binary image with the boundary pixels removed.
    """
    # Create a kernel of ones of shape (pixels, pixels)
    kernel = np.ones((pixels, pixels), np.uint8)

    # Make a new mask where the border is included
    new_binary_mask = binary_mask.copy()
    new_binary_mask[new_binary_mask == 2] = 1

    # Erode the image
    eroded = cv2.erode(new_binary_mask, kernel, iterations=1)

    # If you erode past the area you KNOW is foreground, set it back to 1.
    eroded[binary_mask == 1] = 1
    
    return eroded


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


def square_pad(img):
    if img.shape[0] != img.shape[1]:
        pad = abs(img.shape[0] - img.shape[1]) // 2
        if img.shape[0] > img.shape[1]:
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img


def open_ppm_gz(file_path):
    # Open the gzip file
    with gzip.open(file_path, 'rb') as f:
        # Decompress and read the content
        decompressed_data = f.read()
    # Load the image from the decompressed data
    image = np.array(Image.open(io.BytesIO(decompressed_data)))
    return image


def thunderify_STARE(
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

    # Build a task corresponding to the dataset.
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        subjects = [] 
        # Iterate through the examples.
        for example_name in tqdm(os.listdir(image_root), total=len(list(proc_root.iterdir()))):
            # Define the image_key
            key = example_name.split('.')[0]

            # Paths to the image and segmentation
            img_dir = proc_root / "images" / example_name 

            # Load the image and segmentation.
            img = open_ppm_gz(img_dir)
            mask_dict = {}
            # Iterate through each set of ground truth
            for annotator in ["ah", "vk"]:
                seg_dir = proc_root / f"{annotator}_labels" / example_name.replace('.ppm.gz', f'.{annotator}.ppm.gz')
                mask_dict[annotator] = open_ppm_gz(seg_dir)
            # We also want to make a combined pixelwise-average mask of the two annotators. 
            mask_dict["average"] = np.mean([mask_dict["ah"], mask_dict["vk"]], axis=0)

            # Resize the image and segmentation to config["resize_to"]xconfig["resize_to"]
            img = resize_with_aspect_ratio(img, target_size=config["resize_to"])
            plt.imshow(img)
            plt.show()
            # Next we have to go through the masks and square them.
            gt_prop_dict = {}
            for mask_key in mask_dict:
                # 1. First we squrare pad the mask.
                square_mask = square_pad(mask_dict[mask_key])
                # 2. We record the ground-truth proportion of the mask.
                gt_prop_dict[mask_key] = np.count_nonzero(square_mask) / square_mask.size
                # 3 We then blur the mask a bit. to get the edges a bit more fuzzy.
                smooth_mask = cv2.GaussianBlur(square_mask, (7, 7), 0)
                # 4. We reize the mask to get to our target resolution.
                resized_mask = resize_with_aspect_ratio(smooth_mask, target_size=config["resize_to"])
                # 5. Finally, we normalize it to be [0,1].
                norm_mask = (resized_mask - resized_mask.min()) / (resized_mask.max() - resized_mask.min())
                plt.imshow(norm_mask, cmap='gray')
                plt.show()
                # 6. Store it in the mask dict.
                mask_dict[mask_key] = norm_mask.astype(np.float32)
            
            # Final cleanup steps. 
            img = img.transpose(2, 0, 1).astype(np.float32)
            # Move the channel axis to the front and normalize the labelmap to be between 0 and 1
            assert img.shape == (3, config["resize_to"], config["resize_to"]), f"Image shape isn't correct, got {img.shape}"

            # Save the datapoint to the database
            db[key] = {
                "image": img,
                "masks": mask_dict,
                "gt_props": gt_prop_dict
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
            dataset="STARE",
            version=version,
            resolution=config["resize_to"],
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = splits
        db["_attrs"] = attrs
