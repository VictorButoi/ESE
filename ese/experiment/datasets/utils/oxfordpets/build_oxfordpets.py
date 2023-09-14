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


def proc_OxfordPets(
        data_root, 
        version,
        show=False, 
        save=False
        ):

    proc_root = pathlib.Path(f"/storage/vbutoi/datasets/OxfordPets/processed/{version}") 
    img_root = proc_root / "images"
    mask_root = proc_root / "annotations/trimaps"

    for subj in img_root.iterdir():
        
             

        if save:
            save_root = proc_root / example.name 

            np.save(img_dir, normalized_img_vol)
            np.save(label_dir, rotated_mask_vol)

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


def thunderify_oxfordpets(
        proc_root, 
        dst,
        version
        ):

    # Append version to our paths
    proc_root = proc_root / version
    dst = dst / version

    # Train Calibration Val Test
    splits_ratio = (0.6, 0.1, 0.2, 0.1)
    splits_seed = 42


    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dc_dst), "c") as db:

        # Key track of the ids
        examples = []

        # Iterate through the examples.
        for example in axis_dir.iterdir()

            # Example name
            key = subj.name

            # Paths to the image and segmentation
            img_dir = subj / "image.npy"
            seg_dir = mask_dir / "label.npy"

            # Load the image and segmentation.
            img = np.load(img_dir) 
            seg = np.load(seg_dir)
            
            # Save the datapoint to the database
            db[key] = (img, seg) 
           
            examples.append(key)

        examples = sorted(examples)
        splits = data_splits(subjects, splits_ratio, splits_seed)
        splits = dict(zip(("train", "cal", "val", "test"), splits))

        # Save the metadata
        db["_examples"] = examples 
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="OxfordPets",
            version=version,
        )
        db["_samples"] = examples 
        db["_splits"] = splits
        db["_attrs"] = attrs
