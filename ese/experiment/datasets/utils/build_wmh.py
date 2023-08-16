import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import nibabel.processing as nip
import pathlib
import numpy as np
import os
from thunderpack import ThunderDB

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
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return normalized_image


def proc_WMH(
        data_dirs, 
        show=False, 
        save=False
        ):

    proc_root = pathlib.Path("/storage/vbutoi/datasets/WMH/processed") 

    for ud in data_dirs:

        split_args = str(ud).split("/")
        datacenter = split_args[-2]

        for subj in ud.iterdir():
            print(subj)
            image_dict = {}
            # Get the slices for each modality.
            image_dir = subj / pathlib.Path(f"pre/FLAIR.nii.gz")
            img_volume = resample_nib(nib.load(image_dir))
            # Rotate the volume to be properly oriented
            rotated_volume = np.rot90(img_volume.get_fdata(), k=3, axes=(2, 0))
            # Normalize percentile
            lower = np.percentile(rotated_volume[rotated_volume>0], q=0.5)
            upper = np.percentile(rotated_volume[rotated_volume>0], q=99.5)
            clipped_volume = np.clip(rotated_volume, a_min=lower, a_max=upper)
            # Make the image square
            sqr_img = pad_image_numpy(clipped_volume, (256, 256, 256))
            # Store in the dictionary
            image_dict["img"] = sqr_img  

            # Get the label slice
            if "training" in str(ud):
                label_versions = ["observer_o12", "observer_o3", "observer_o4"]
            else:
                label_versions = ["observer_o12"]

            image_dict["segs"] = {}
            for annotator_name in label_versions:

                # Need to acocunt for extra annotators
                if annotator_name != "observer_o12":
                    split_name = str(ud).split("WMH")
                    alternate_seg = pathlib.Path(split_name[0] + f"/WMH/additional_annotations/{annotator_name}/" + split_name[1] + f"/{subj.name}")
                    seg_dir = alternate_seg / "result.nii.gz"
                else:
                    seg_dir = subj / "wmh.nii.gz"

                seg = resample_mask_to(nib.load(seg_dir), img_volume)
                # Rotate the volume to be properly oriented
                rotated_seg = np.rot90(seg.get_fdata(), k=3, axes=(2, 0))
                # Do the sliceing
                binary_seg = np.uint8(rotated_seg== 1)
                # Make the image square
                sqr_seg = pad_image_numpy(binary_seg, (256, 256, 256))
                # Store in dictionary
                image_dict["segs"][annotator_name] = sqr_seg 
            
            if show:
                # Plot the slices
                num_segs = len(image_dict["segs"].keys())
                im_vol = image_dict["img"]
                for axis in [0, 1, 2]:
                    f, axarr = plt.subplots(1, num_segs + 1, figsize=(5 * (num_segs + 1), 5))

                    # Do the sliceing
                    mid_axis_slice = im_vol.shape[axis] // 2 
                    img_slice = np.take(im_vol, mid_axis_slice, axis=axis)
                    
                    # Show the image
                    im = axarr[0].imshow(img_slice, cmap='gray')
                    axarr[0].set_title("Image")
                    f.colorbar(im, ax=axarr[0], orientation='vertical') 
                    
                    # Show the segs
                    for an_idx, annotator in enumerate(image_dict["segs"].keys()):
                        seg_vol = image_dict["segs"][annotator]
                        seg_slice = np.take(seg_vol, mid_axis_slice, axis=axis)
                        im = axarr[an_idx + 1].imshow(seg_slice, cmap="gray")
                        axarr[an_idx + 1].set_title(annotator)
                        f.colorbar(im, ax=axarr[an_idx + 1], orientation='vertical')
                    plt.show()  
            
            if save:
                save_root = proc_root / datacenter / subj.name 
                if not save_root.exists():
                    os.makedirs(save_root)
                
                # Save your image
                img_dir = save_root / "image.npy"
                np.save(img_dir, image_dict["img"])

                for annotator in image_dict["segs"].keys():
                    # This is how we organize the data.
                    annotator_dir = save_root / annotator 
                    if not annotator_dir.exists():
                        os.makedirs(annotator_dir)

                    label_dir = annotator_dir / "label.npy"
                    np.save(label_dir, image_dict["segs"][annotator])


VERSION = "v0.1"

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


def thunderify_wmh(proc_root, dst):
    datacenters = proc_root.iterdir() 

    # Train Calibration Val Test
    splits_ratio = (0.6, 0.1, 0.2, 0.1)
    splits_seed = 42

    for dc in datacenters:
        dc_proc_dir = proc_root / dc.name
        dc_dst = dst / dc.name
        with ThunderDB.open(str(dc_dst), "c") as db:
            subjects = []
            num_annotators = []
            subj_list = dc_proc_dir.iterdir()
            for subj in subj_list:
                key = subj.name
                print(subj)
                img_dir = subj / "image.npy"
                img = np.load(img_dir) 
                mask_list = list(subj.glob("observer*"))
                mask_dict = {}
                for mask_dir in mask_list:
                    seg_dir = mask_dir / "label.npy"
                    seg = np.load(seg_dir)
                    mask_dict[mask_dir.name] = seg
                db[key] = {
                    "image": img,
                    "masks": mask_dict
                }
                subjects.append(key)
                num_annotators.append(len(mask_list))

            subjects, num_annotators = zip(*sorted(zip(subjects, num_annotators)))
            splits = data_splits(subjects, splits_ratio, splits_seed)
            splits = dict(zip(("train", "cal", "val", "test"), splits))
            db["_subjects"] = subjects
            db["_splits"] = splits
            db["_splits_kwarg"] = {
                "ratio": splits_ratio, 
                "seed": splits_seed
                }
            attrs = dict(
                dataset="WMH",
                version=VERSION,
                group=dc.name,
                modality="FLAIR",
                resolution=256,
            )
            db["_num_annotators"] = num_annotators 
            db["_subjects"] = subjects
            db["_samples"] = subjects
            db["_splits"] = splits
            db["_attrs"] = attrs