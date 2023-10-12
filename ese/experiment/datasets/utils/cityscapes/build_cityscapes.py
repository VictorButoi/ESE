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


def proc_CityScapes(
        cfg: Config
        ):

    config = cfg.to_dict()

    # Where the data is 
    data_root = pathlib.Path(config['data_root'])
    img_root = data_root / "leftImg8bit"
    mask_root = data_root / "gtFine"

    # This is where we will save the processed data
    proc_root = data_root / "processed" / str(config['version'])

    ex_counter = 0
    for split_dir in tqdm(img_root.iterdir(), total=len(list(img_root.iterdir()))):
        for city_center_dir in tqdm(split_dir.iterdir(), total=len(list(split_dir.iterdir()))):
            for example_dir in city_center_dir.iterdir():
                # Read the image and label
                try:
                    label_dir = mask_root / split_dir.name / city_center_dir.name / example_dir.name.replace("leftImg8bit", "gtFine_labelIds")

                    img = np.array(Image.open(example_dir))
                    label = np.array(Image.open(label_dir))

                    if config["show_examples"]:
                        f, axarr = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

                        # Show the original label
                        im = axarr[0].imshow(img, interpolation='None')
                        axarr[0].set_title("Image")
                        f.colorbar(im, ax=axarr[0])
                        
                        # Show thew new label
                        axarr[1].imshow(img)
                        nl = axarr[1].imshow(label, alpha=0.5, interpolation='None')
                        axarr[1].set_title("Image + Mask")
                        f.colorbar(nl, ax=axarr[1])

                        # Show thew new label
                        lb = axarr[2].imshow(label, interpolation='None')
                        axarr[2].set_title("Mask Only")
                        f.colorbar(lb, ax=axarr[2])
                        
                        plt.show()

                        if ex_counter > config["num_examples_to_show"]:   
                            break
                        # Only count examples if showing examples
                        ex_counter += 1

                    if config["save"]:
                        example_name = "_".join(example_dir.name.split("_")[:-1])
                        save_root = proc_root / split_dir.name/ example_name
                        
                        if not save_root.exists():
                            save_root.mkdir(parents=True)

                        img_save_dir = save_root / "image.npy"
                        label_save_dir = save_root / "label.npy"

                        np.save(img_save_dir, img)
                        np.save(label_save_dir, label)

                except Exception as e:
                    print(f"Error with {example_dir.name}: {e}. Skipping")

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


def thunderify_OxfordPets(
        proc_root, 
        dst,
        version
        ):

    # Append version to our paths
    proc_root = proc_root / version
    dst_dir = dst / version

    # Append version to our paths
    splits_ratio = (0.6, 0.1, 0.2, 0.1)
    splits_seed = 42

    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:

        # Key track of the ids
        examples = []

        # Iterate through the examples.
        for example in tqdm(proc_root.iterdir(), total=len(list(proc_root.iterdir()))):

            # Example name
            key = example.name

            # Paths to the image and segmentation
            img_dir = example / "image.npy"
            seg_dir = example / "label.npy"

            try:
                # Load the image and segmentation.
                img = np.load(img_dir)
                img = img.transpose(2, 0, 1)
                seg = np.load(seg_dir)
                
                # Convert to the right type
                img = img.astype(np.float32)
                seg = seg.astype(np.int64)

                assert img.shape == (3, 256, 256), f"Image shape is {img.shape}"
                assert seg.shape == (256, 256), f"Seg shape is {seg.shape}"
                assert np.count_nonzero(seg) > 0, "Label can't be empty."
                
                # Save the datapoint to the database
                db[key] = (img, seg) 
                
                examples.append(key)
            
            except Exception as e:
                print(f"Error with {key}: {e}. Skipping")

        examples = sorted(examples)
        classes = ["_".join(ex.split("_")[:-1]) for ex in examples]
        data_ids = [ex.split("_")[-1] for ex in examples]

        splits = data_splits(examples, splits_ratio, splits_seed)
        splits = dict(zip(("train", "cal", "val", "test"), splits))

        # Save the metadata
        db["_examples"] = examples 
        db["_samples"] = examples 
        db["_classes"] = classes 
        db["_ids"] = data_ids 
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="OxfordPets",
            version=version,
        )
        db["_splits"] = splits
        db["_attrs"] = attrs
