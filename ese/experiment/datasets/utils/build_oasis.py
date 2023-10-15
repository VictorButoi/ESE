import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nibabel.processing as nip
from ionpy.util import Config
import pathlib
import numpy as np
import os
from tqdm import tqdm
from thunderpack import ThunderDB
from scipy.ndimage import zoom

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    return new_img


def resample_mask_to(msk, to_img):
    """Resamples the nifti mask from its original spacing to a new spacing specified by its corresponding image
    
    Parameters:
    ----------
    msk: The nibabel nifti mask to be resampled
    to_img: The nibabel image that acts as a template for resampling
    
    Returns:
    ----------
    new_msk: The resampled nibabel mask 
    
    """
    to_img.header['bitpix'] = 8
    to_img.header['datatype'] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)
    return new_msk


def pad_image_numpy(arr, target_dims):
    orig_shape = arr.shape
    pad_width = [(max((t - o) // 2, 0), max((t - o + 1) // 2, 0)) for o, t in zip(orig_shape, target_dims)]
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=arr[10,10,10])
    return padded_arr 


def normalize_image(image):
    # Convert the image to floating point format
    image = image.astype(np.float32)

    # Normalize the image between 0 and 1
    val_range = np.max(image) - np.min(image)
    # If the image is a constant, just return a zero image
    if val_range == 0:
        normalized_image = np.zeros(image.shape)
    else:
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return normalized_image


def normalize_volume(volume):
    return np.stack([normalize_image(volume[slice_idx, ...]) for slice_idx in range(volume.shape[0])], axis=0)


def rotate_y_axis(array):
    array_rot_y = np.moveaxis(array, (0, 1, 2), (2, 1, 0))
    array_rot_y = np.rot90(array_rot_y, axes=(1, 2))
    return array_rot_y


def proc_OASIS(
    cfg: Config     
    ):
    # Where the data is
    d_root = pathlib.Path(cfg["data_root"])
    proc_root = d_root / "processed" / str(cfg['version'])
    example_dir = d_root / "raw_files"
    # This is where we will save the processed data
    for subj in tqdm(example_dir.iterdir(), total=len(list(example_dir.iterdir()))):
        ####################################
        # Image
        ####################################
        # Get the slices for each modality.
        image_dir = subj / "aligned_norm.nii.gz"
        img_ngz = resample_nib(nib.load(image_dir))
        img_numpy = img_ngz.get_fdata()
        # Rotate the volume to be properly oriented
        # rotated_volume = np.rot90(img_volume.get_fdata(), k=3, axes=(2, 0))
        # Make the image square
        max_img_dim = max(img_numpy.shape)
        sqr_img = pad_image_numpy(img_numpy, (max_img_dim, max_img_dim, max_img_dim))
        # Resize to 256
        zoom_factors = np.array([256, 256, 256]) / np.array(sqr_img.shape)
        resized_img = zoom(sqr_img, zoom_factors, order=1)  # You can adjust the 'order' parameter for interpolation quality
        # Make the type compatible
        resized_img = resized_img.astype(np.float32)
        # Norm the iamge volume
        norm_img_vol = normalize_image(resized_img)
        # Reshape by flipping z
        flipped_img_vol = np.flip(norm_img_vol, axis=2)
        processed_img_vol = np.rot90(flipped_img_vol, k=3, axes=(0,2))
        for z in range(processed_img_vol.shape[2]):
            processed_img_vol[:, :, z] = np.rot90(processed_img_vol[:, :, z], 3)

        ####################################
        # Segmentation 
        ####################################
        # Get the label slice
        seg35_dir = subj / "aligned_seg35.nii.gz"
        seg4_dir = subj / "aligned_seg4.nii.gz"

        def process_OASIS_seg(label_dir, ref_image):
            seg_ngz = resample_mask_to(nib.load(label_dir), ref_image)
            seg_npy = seg_ngz.get_fdata()
            # Make the image square
            max_seg_dim = max(seg_npy.shape)
            sqr_seg = pad_image_numpy(seg_npy, (max_seg_dim, max_seg_dim, max_seg_dim))
            # Resize to 256
            zoom_factors = np.array([256, 256, 256]) / np.array(sqr_seg.shape)
            resized_seg = zoom(sqr_seg, zoom_factors, order=0)  # You can adjust the 'order' parameter for interpolation quality
            # Reshape by flipping z
            flipped_seg_vol = np.flip(resized_seg, axis=2)
            rot_seg_vol = np.rot90(flipped_seg_vol, k=3, axes=(0,2))
            for z in range(rot_seg_vol.shape[2]):
                rot_seg_vol[:, :, z] = np.rot90(rot_seg_vol[:, :, z], 3)
            return rot_seg_vol
        
        processed_seg35_vol = process_OASIS_seg(seg35_dir, img_ngz)
        processed_seg4_vol = process_OASIS_seg(seg4_dir, img_ngz)

        if cfg['show_examples']:
            # Set up the figure
            f, axarr = plt.subplots(3, 3, figsize=(15, 10))
            # Plot the slices
            for major_axis in [0, 1, 2]:
                # Figure out how to spin the volume.
                all_axes = [0, 1, 2]
                all_axes.remove(major_axis)
                tranposed_axes = tuple([major_axis] + all_axes)
                # Spin the volumes 
                axis_img_vol = np.transpose(processed_img_vol, tranposed_axes)
                axis_seg35_vol = np.transpose(processed_seg35_vol, tranposed_axes)
                axis_seg4_vol = np.transpose(processed_seg4_vol, tranposed_axes)

                # Do the slicing
                img_slice = axis_img_vol[128, ...]
                # Show the image
                im = axarr[0, major_axis].imshow(img_slice, cmap='gray')
                axarr[0, major_axis].set_title(f"Image Axis: {major_axis}")
                f.colorbar(im, ax=axarr[0, major_axis], orientation='vertical') 

                # Show the segs
                seg35_slice = axis_seg35_vol[128, ...]
                im = axarr[1, major_axis].imshow(seg35_slice, cmap="tab20b")
                axarr[1, major_axis].set_title(f"35 Lab Seg Axis: {major_axis}")
                f.colorbar(im, ax=axarr[1, major_axis], orientation='vertical')

                seg4_slice = axis_seg4_vol[128, ...]
                im = axarr[2, major_axis].imshow(seg4_slice, cmap="tab20b")
                axarr[2, major_axis].set_title(f"4 Lab Seg Axis: {major_axis}")
                f.colorbar(im, ax=axarr[2, major_axis], orientation='vertical')
            plt.show()  

        if cfg['save']:
            for major_axis in [0, 1, 2]:
                save_root = proc_root / str(major_axis) / subj.name 
                if not save_root.exists():
                    os.makedirs(save_root)
                # Figure out how to spin the volume.
                all_axes = [0, 1, 2]
                all_axes.remove(major_axis)
                tranposed_axes = tuple([major_axis] + all_axes)
                # Spin the volumes 
                axis_img_vol = np.transpose(processed_img_vol, tranposed_axes)
                axis_seg35_vol = np.transpose(processed_seg35_vol, tranposed_axes)
                axis_seg4_vol = np.transpose(processed_seg4_vol, tranposed_axes)
                # Save your image
                img_dir = save_root / "image.npy"
                label35_dir = save_root / "label35.npy"
                label4_dir = save_root / "label4.npy"
                # This is how we organize the data.
                np.save(img_dir, axis_img_vol)
                np.save(label35_dir, axis_seg35_vol)
                np.save(label4_dir, axis_seg4_vol)


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


def thunderify_OASIS(
        cfg: Config
        ):
    # Append version to our paths
    proc_root = pathlib.Path(cfg["proc_root"]) / str(cfg['version'])
    thunder_dst = pathlib.Path(cfg["dst_dir"]) / str(cfg['version'])
    # Train Calibration Val Test
    splits_ratio = (0.7, 0.1, 0.1, 0.1)
    splits_seed = 42
    # Iterate through all axes and the two different labeling protocols.
    for axis_examples_dir in proc_root.iterdir():
        for label_set in ["label35", "label4"]:
            task_save_dir = thunder_dst / axis_examples_dir.name / label_set
            # Make the save dir if it doesn't exist
            if not task_save_dir.exists():
                os.makedirs(task_save_dir)
            # Iterate through each datacenter, axis  and build it as a task
            with ThunderDB.open(str(task_save_dir), "c") as db:
                subjects = []
                for subj in axis_examples_dir.iterdir():
                    # Load the image
                    img_dir = subj / "image.npy"
                    raw_img = np.load(img_dir) 
                    #Load the label
                    lab_dir = subj / f"{label_set}.npy"
                    raw_lab = np.load(lab_dir)
                    # Convert the img and label to correct types
                    img = raw_img.astype(np.float32)
                    lab = raw_lab.astype(np.int64)
                    # Save the datapoint to the database
                    key = subj.name
                    db[key] = (img, lab) 
                    subjects.append(key)
                # Sort the subjects and save some info.
                subjects = sorted(subjects)
                splits = data_splits(subjects, splits_ratio, splits_seed)
                splits = dict(zip(("train", "cal", "val", "test"), splits))
                db["_subjects"] = subjects
                db["_splits"] = splits
                db["_splits_kwarg"] = {
                    "ratio": splits_ratio, 
                    "seed": splits_seed
                    }
                attrs = dict(
                    dataset="OASIS",
                    version=cfg['version'],
                    label_set=label_set,
                    axis=axis_examples_dir.name
                )
                db["_subjects"] = subjects
                db["_samples"] = subjects
                db["_splits"] = splits
                db["_attrs"] = attrs
