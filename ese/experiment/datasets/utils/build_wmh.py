import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import nibabel.processing as nip
import pathlib
import numpy as np
import os
from PIL import Image

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


def pad_image_numpy(image_array, target_width, target_height):
    # Convert the NumPy array to a Pillow image
    original_image = Image.fromarray(image_array)

    # Get the original dimensions
    original_width, original_height = original_image.size

    # Calculate the amount of padding needed
    width_padding = target_width - original_width
    height_padding = target_height - original_height

    # Calculate the padding borders
    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding

    # Get the border pixel value
    border_pixel = original_image.getpixel((10, 10))

    # Create a new image with the desired dimensions and fill with border pixel
    padded_image = Image.new(original_image.mode, (target_width, target_height), border_pixel)

    # Paste the original image onto the padded image with padding borders
    padded_image.paste(original_image, (left_padding, top_padding))

    # Convert the padded image back to a NumPy array
    padded_image_array = np.array(padded_image)

    return padded_image_array


def normalize_image(image):
    # Convert the image to floating point format
    image = image.astype(np.float32)

    # Normalize the image between 0 and 1
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return normalized_image


def proc_WMH(
        data_dirs, 
        modalities, 
        show=False, 
        save=False
        ):

    proc_root = pathlib.Path("/storage/vbutoi/datasets/WMH/processed") 

    for ud in data_dirs:
        print(ud)
        split_args = str(ud).split("/")
        datacenter = split_args[-2]
        modality = split_args[-1]
        for subj in ud.iterdir():
            for axis in [0, 1, 2]:
                image_dict = {}
                # Get the slices for each modality.
                for mod_idx, modality in enumerate(modalities):
                    image_dir = subj / pathlib.Path(f"pre/{modality}.nii.gz")
                    print("Loading image: ", image_dir)
                    img_volume = resample_nib(nib.load(image_dir))
                    mid_axis_slice = img_volume.get_fdata().shape[axis] // 2 
                    img_modality_slice = np.take(img_volume.get_fdata(), mid_axis_slice, axis=axis)
                    # Normalize percentile
                    lower = np.percentile(img_modality_slice[img_modality_slice>0], q=0.5)
                    upper = np.percentile(img_modality_slice[img_modality_slice>0], q=99.5)
                    clipped_image = np.clip(img_modality_slice, a_min=lower, a_max=upper)
                    # Make the image square
                    sqr_img = pad_image_numpy(clipped_image, 256, 256)
                    # Normalize the now square image
                    norm_img = normalize_image(sqr_img)
                    # Fix the orientation
                    rotated_sqr_img = np.rot90(norm_img, k=3, axes=(1, 0))
                    # Store in the dictionary
                    image_dict[modality] = rotated_sqr_img

                # Get the label slice
                if "training" in str(ud):
                    label_versions = ["observer_o12", "observer_o3", "observer_o4"]
                else:
                    label_versions = ["observer_o12"]

                image_dict["segs"] = {}
                for annotator_name in label_versions:
                    if annotator_name != "observer_o12":
                        split_name = str(ud).split("WMH")
                        alternate_seg = pathlib.Path(split_name[0] + f"/WMH/additional_annotations/{annotator_name}/" + split_name[1] + f"/{subj.name}")
                        seg_dir = alternate_seg / "result.nii.gz"
                    else:
                        seg_dir = subj / "wmh.nii.gz"
                    print("Loading seg: ", seg_dir)
                    seg = resample_mask_to(nib.load(seg_dir), img_volume)
                    seg_slice = np.take(seg.get_fdata(), mid_axis_slice, axis=axis)
                    binary_seg_slice = np.uint8(seg_slice == 1)
                    sqr_seg = pad_image_numpy(binary_seg_slice, 256, 256)
                    rotated_sqr_seg = np.rot90(sqr_seg, k=3, axes=(1, 0))
                    image_dict["segs"][annotator_name] = rotated_sqr_seg 
                
                if show:
                    # Plot the slices
                    num_segs = len(image_dict["segs"].keys())
                    f, axarr = plt.subplots(1, num_segs + 1, figsize=(5 * (num_segs + 1), 5))
                    im = axarr[0].imshow(image_dict["FLAIR"], cmap='gray')
                    axarr[0].set_title("Image")
                    f.colorbar(im, ax=axarr[0], orientation='vertical') 
                    for an_idx, annotator in enumerate(image_dict["segs"].keys()):
                        im = axarr[an_idx + 1].imshow(image_dict["segs"][annotator], cmap="gray")
                        axarr[an_idx + 1].set_title(annotator)
                        f.colorbar(im, ax=axarr[an_idx + 1], orientation='vertical')
                    plt.show()  
                
                if save:
                    save_root = proc_root / datacenter / str(axis) / subj.name 
                    if not save_root.exists():
                        os.makedirs(save_root)
                    
                    # Save your image
                    img_dir = save_root / "image.npy"
                    print("Saving Image: ", img_dir)
                    np.save(img_dir, image_dict[modality])

                    for annotator in image_dict["segs"].keys():
                        # This is how we organize the data.
                        annotator_dir = save_root / annotator 
                        if not annotator_dir.exists():
                            os.makedirs(annotator_dir)

                        label_dir = annotator_dir / "label.npy"
                        print("Saving Label: ", label_dir)
                        np.save(label_dir, image_dict["segs"][annotator])