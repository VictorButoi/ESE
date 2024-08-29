import os
import cv2
import time
import matplotlib.pyplot as plt
import pathlib
import voxel as vx
from tqdm import tqdm
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
    proc_root = pathlib.Path(config["root"]) / 'raw_data' / 'organized_data'
    dst_dir = pathlib.Path(config["dst_folder"]) / version

    isl_raw_root = proc_root / 'raw_data'
    isl_prc_root = proc_root / 'derivatives'

    # Iterate through each datacenter, axis  and build it as a task
    # with ThunderDB.open(str(dst_dir), "c") as db:
    # Key track of the ids
    subjects = [] 
    # Iterate through the examples.
    subj_list = list(os.listdir(isl_raw_root))
    for subj_name in tqdm(os.listdir(isl_raw_root), total=len(subj_list)):

        # Paths to the image and segmentation
        img_dir = isl_raw_root / subj_name / 'ses-02' / f'{subj_name}_ses-02_dwi.nii.gz' 
        seg_dir = isl_prc_root / subj_name / 'ses-02' / f'{subj_name}_ses-02_lesion-msk.nii.gz'

        # Load the image and segmentation.
        print(vx.__file__)
        vol = vx.load_volume(img_dir)
        print(vol.geometry.spacing)

        raise ValueError

        # We need to normalize our volume to be between 0 and 1.
        raw_img_volume = (raw_img_volume - raw_img_volume.min()) / (raw_img_volume.max() - raw_img_volume.min())

        # Visualize the slice corresponding to most amount of label in raw_seg
        # lab_amounts = croppped_seg_volume.sum(axis=(0, 1))
        # max_slice_idx = np.argmax(lab_amounts)
        midslice = cropped_img_volume.shape[2] // 2 
        cropped_img = croppped_img_volume[:, :, midslice]
        cropped_seg = croppped_seg_volume[:, :, midslice]
        # Visualize using matpltolib
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        im = ax[0].imshow(cropped_img, cmap='gray')
        sg = ax[1].imshow(cropped_seg, cmap='gray')
        # Add colorbars
        fig.colorbar(im, ax=ax[0])
        fig.colorbar(sg, ax=ax[1])
        plt.show()
        time.sleep(2)


    #     # For this dataset, as it is non-binary, we have to do something more clever.
    #     # First we need to map our segmentation to a binary segmentation.
    #     unique_labels = np.unique(raw_seg)
    #     mask_dict = {} 
    #     gt_prop_dict = {}

    #     # If the unique labels are greater than 2, then we have to do some work.
    #     square_img = square_pad(raw_img)
    #     square_raw_seg = square_pad(raw_seg)

    #     # Remove the axes and ticks
    #     if square_img.shape[0] != config["resize_to"]:
    #         square_img = resize_with_aspect_ratio(square_img, target_size=config["resize_to"]).astype(np.float32)

    #     for lab in unique_labels:
    #         # If the lab != 0, then we make a binary mask.
    #         if lab != 0:
    #             # Make a binary mask.
    #             binary_mask = (square_raw_seg == lab).astype(np.float32)

    #             # Get the proportion of the binary mask.
    #             gt_prop = np.count_nonzero(binary_mask) / binary_mask.size
    #             gt_prop_dict[lab] = gt_prop

    #             # If we are resizing then we need to smooth the mask.
    #             if square_raw_seg.shape[0] != config["resize_to"]:
    #                 smooth_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
    #                 # Now we process the mask in our standard way.
    #                 proc_mask = resize_with_aspect_ratio(smooth_mask, target_size=config["resize_to"])
    #             else:
    #                 proc_mask = binary_mask

    #             # Renormalize the mask to be between 0 and 1.
    #             norm_mask = (proc_mask - proc_mask.min()) / (proc_mask.max() - proc_mask.min())

    #             # Store the mask in the mask_dict.
    #             mask_dict[lab] = norm_mask.astype(np.float32)
            
    #     assert square_img.shape == (config["resize_to"], config["resize_to"]), f"Image shape isn't correct, got {square_img.shape}"

    #     # Save the datapoint to the database
    #     # db[key] = {
    #     #     "img": square_img, 
    #     #     "seg": mask_dict,
    #     #     "gt_proportion": gt_prop_dict 
    #     # } 
    #     subjects.append(key)

    # subjects = sorted(subjects)
    # splits = data_splits(subjects, splits_ratio, splits_seed)
    # splits = dict(zip(("train", "cal", "val", "test"), splits))
    # print(splits)
    # # for split_key in splits:
    # #     print(f"{split_key}: {len(splits[split_key])} samples")

    # # # Save the metadata
    # # db["_subjects"] = subjects
    # # db["_splits"] = splits
    # # db["_splits_kwarg"] = {
    # #     "ratio": splits_ratio, 
    # #     "seed": splits_seed
    # #     }
    # # attrs = dict(
    # #     dataset="OCTA_6M",
    # #     version=version,
    # #     resolution=config["resize_to"],
    # # )
    # # db["_subjects"] = subjects
    # # db["_samples"] = subjects
    # # db["_splits"] = splits
    # # db["_attrs"] = attrs

    