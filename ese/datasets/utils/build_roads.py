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

# Local imports
from .utils_for_build import data_splits, normalize_image


def thunderify_Roads(
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

    # # Iterate through each datacenter, axis  and build it as a task
    # with ThunderDB.open(str(dst_dir), "c") as db:
    # Key track of the ids
    subjects = [] 
    # Iterate through the examples.
    subj_list = list(os.listdir(image_root))

    # I want to keep two files open, one for good examples and one for bad examples.
    # I will write the bad examples to a file and then remove them from the dataset.
    good_ex_file = "/storage/vbutoi/datasets/Roads/assets/good_examples.txt"
    bad_ex_file = "/storage/vbutoi/datasets/Roads/assets/bad_examples.txt"
    counter = 0

    # Open both files
    with open(good_ex_file, "r+") as good_f, open(bad_ex_file, "r+") as bad_f:
        good_example_list = [line.strip() for line in good_f]  # Reads and strips newline characters
        bad_examples_list = [line.strip() for line in bad_f]  # Reads and strips newline characters

        for example_name in tqdm(os.listdir(image_root), total=len(subj_list)):
            counter += 1

            if example_name not in good_example_list and example_name not in bad_examples_list:
                # Define the image_key
                key = "subject_" + example_name.split('_')[0]

                # Paths to the image and segmentation
                img_dir = proc_root / "images" / example_name 
                seg_dir = proc_root / "masks" / example_name.replace("tiff", "tif")

                # Load the image and segmentation.
                img = np.array(Image.open(img_dir))
                seg = np.array(Image.open(seg_dir))

                # Normalize the seg to be between 0 and 1
                seg = normalize_image(seg)
                print("Example: ", example_name)

                # Get the proportion of the binary mask.
                gt_prop = np.count_nonzero(seg) / seg.size

                # Visualize the image and mask
                # if config.get("visualize", False):
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                im = ax[0].imshow(img)
                ax[0].set_title("Image")
                fig.colorbar(im, ax=ax[0])
                se = ax[1].imshow(seg, cmap="gray")
                fig.colorbar(se, ax=ax[1])
                ax[1].set_title("Mask")
                plt.show()

                # Move the last channel of image to the first channel
                img = np.moveaxis(img, -1, 0)
                seg = seg[np.newaxis, ...]

                # Prompt the user to decide if the image is good or not.
                is_good = input("Is this image good? (y/n): ")
                if is_good.lower() == "n":
                    print("logged bad example:", example_name)
                    # Write the bad example to the file.
                    bad_f.write(example_name + "\n")
                else:
                    print("logged good example:", example_name)
                    # Write the good example to the file.
                    good_f.write(example_name + "\n")

        #     # Save the datapoint to the database
        #     db[key] = {
        #         "img": img, 
        #         "seg": seg,
        #         "gt_proportion": gt_prop 
        #     } 
        #     subjects.append(key)

        # subjects = sorted(subjects)
        # splits = data_splits(subjects, splits_ratio, splits_seed)
        # splits = dict(zip(("train", "cal", "val", "test"), splits))
        # for split_key in splits:
        #     print(f"{split_key}: {len(splits[split_key])} samples")

        # # Save the metadata
        # db["_subjects"] = subjects
        # db["_splits"] = splits
        # db["_splits_kwarg"] = {
        #     "ratio": splits_ratio, 
        #     "seed": splits_seed
        #     }
        # attrs = dict(
        #     dataset="Roads",
        #     version=version,
        # )
        # db["_subjects"] = subjects
        # db["_samples"] = subjects
        # db["_splits"] = splits
        # db["_attrs"] = attrs

        