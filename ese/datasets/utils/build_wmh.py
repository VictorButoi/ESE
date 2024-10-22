import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import nibabel.processing as nip
import pathlib
import numpy as np
import os
from thunderpack import ThunderDB
from scipy.ndimage import zoom

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def thunderify_legacy_WMH(
    proc_root, 
    dst,
    version
):

    # Append version to our paths
    proc_root = proc_root / version
    dst = dst / version

    datacenters = proc_root.iterdir() 
    # Train Calibration Val Test
    splits_ratio = (0.6, 0.1, 0.2, 0.1)
    splits_seed = 42

    for dc in datacenters:
        dc_proc_dir = proc_root / dc.name
        for axis_dir in dc_proc_dir.iterdir():
            dc_dst = dst / dc.name / axis_dir.name
            print(str(dc_dst))
            if not dc_dst.exists():
                os.makedirs(dc_dst)

            # Iterate through each datacenter, axis  and build it as a task
            with ThunderDB.open(str(dc_dst), "c") as db:
                subj_list = axis_dir.iterdir()

                subjects = []
                num_annotators = []

                for subj in subj_list:
                    key = subj.name
                    img_dir = subj / "image.npy"

                    img = np.load(img_dir) 
                    mask_list = list(subj.glob("observer*"))

                    mask_dict = {}
                    pp_dict = {}
                    
                    # Iterate through each set of ground truth
                    for mask_dir in mask_list:
                        seg_dir = mask_dir / "label.npy"
                        seg = np.load(seg_dir)
                        
                        # Our processing has ensured that we care about axis 0.
                        pixel_proportion = np.sum(seg, axis=(1,2))
                        mask_dict[mask_dir.name] = seg
                        pp_dict[mask_dir.name] = pixel_proportion

                    # Save the datapoint to the database
                    db[key] = {
                        "image": img,
                        "masks": mask_dict,
                        "pixel_proportions": pp_dict
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
                    version=version,
                    group=dc.name,
                    modality="FLAIR",
                    axis=axis_dir.name,
                    resolution=256,
                )
                db["_num_annotators"] = num_annotators 
                db["_subjects"] = subjects
                db["_samples"] = subjects
                db["_splits"] = splits
                db["_attrs"] = attrs


############################################################################################################
# SERIES OF HELPER FUNCTIONS IMPORTANT FOR SETTING UP WMH

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def gather_all_subject_dirs():
    root = pathlib.Path("/storage/vbutoi/datasets/WMH/original_unzipped")
    splits = ["training", "test", "additional_annotations"]

    all_dirs = []

    for split in splits:
        split_path = root / split
        for subdir in split_path.iterdir():
            print(subdir)
            all_dirs.append(subdir)
            for l3_dir in subdir.iterdir():
                if not is_integer(str(l3_dir.name)):
                    print(l3_dir)
                    all_dirs.append(l3_dir)
                    for l4_dir in l3_dir.iterdir():
                        if not is_integer(str(l4_dir.name)):
                            print(l4_dir)
                            all_dirs.append(l4_dir)
                            for l5_dir in l4_dir.iterdir():
                                if not is_integer(str(l5_dir.name)):
                                    print(l5_dir)
                                    all_dirs.append(l5_dir)
    unique_dirs_with_additional = []
    for path in all_dirs:
        all_other_dirs = [p for p in all_dirs if p != path]
        is_subdir = False
        for other_path in all_other_dirs:
            if path in other_path.parents:
                is_subdir = True
                break
        if not is_subdir:
            unique_dirs_with_additional.append(path)
    
    return unique_dirs_with_additional


# CELL FOR ORGANIZING ORIGINAL DATASET
def organize_WMH_part1(unique_dirs_with_additional):
    unique_annotator_o1_dirs = [ud for ud in unique_dirs_with_additional if "additional_annotations" not in str(ud)]
    organized_dir = "/storage/vbutoi/datasets/WMH/raw_organized"

    for o12_path in unique_annotator_o1_dirs:
        dataset = o12_path.parts[-2]
        print(dataset)
        print(o12_path)
        # Iterate through the subjects
        for subject_dir in o12_path.iterdir():
            subject_id = subject_dir.parts[-1]
            img_dir = subject_dir / "pre" / "FLAIR.nii.gz"
            seg_dir = subject_dir / "wmh.nii.gz"
            print(subject_id, img_dir.exists(), seg_dir.exists())
            # Now we need to make a copy of the image and segmentation in the organized directory
            organized_dataset_dir = os.path.join(organized_dir, dataset)
            new_subject_dir = os.path.join(organized_dataset_dir, subject_id)
            # Assert that the subject directory does not exist
            assert not os.path.exists(new_subject_dir)
            os.makedirs(new_subject_dir)
            new_img_dir = os.path.join(new_subject_dir, "FLAIR.nii.gz")
            new_seg_dir = os.path.join(new_subject_dir, "annotator_o12_mask.nii.gz")
            # Copy the files
            os.system(f"cp {img_dir} {new_img_dir}")
            os.system(f"cp {seg_dir} {new_seg_dir}")


# CELL FOR ORGANIZING ADDITIONAL ANNOTATIONS
def organize_WMH_part2(unique_dirs_with_additional):

    other_anno_dirs = [ud for ud in unique_dirs_with_additional if "additional_annotations" in str(ud)]
    organized_dir = "/storage/vbutoi/datasets/WMH/raw_organized"

    for new_annotator_path in other_anno_dirs:
        dataset = new_annotator_path.parts[-2]
        annotator = new_annotator_path.parts[-4]

        if annotator == 'observer_o3':
            annotator = 'annotator_o3'
        elif annotator == 'observer_o4':
            annotator = 'annotator_o4'
        else:
            raise ValueError(f"Unknown annotator {annotator}")

        print(dataset)
        print(annotator)
        print(new_annotator_path)
        # Iterate through the subjects
        for additional_segs_dir in new_annotator_path.iterdir():
            subject_id = additional_segs_dir.parts[-1]
            alternate_seg_dir = additional_segs_dir / "result.nii.gz"
            print(subject_id, alternate_seg_dir.exists())
            # Now we need to make a copy of the alternative segmentation in the organized directory
            organized_dataset_dir = os.path.join(organized_dir, dataset)
            new_subject_dir = os.path.join(organized_dataset_dir, subject_id)
            # This path should already exist.
            assert os.path.exists(new_subject_dir)
            new_annotator_seg_dir = os.path.join(new_subject_dir, f"{annotator}_mask.nii.gz")

            # Copy the files
            os.system(f"cp {alternate_seg_dir} {new_annotator_seg_dir}")