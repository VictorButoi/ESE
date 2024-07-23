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


def proc_ADE20K(
        cfg: Config,
        num_examples_to_show: int = 10
        ):
    # Get the configk
    config = cfg.to_dict()
    # Where the data is 
    data_root = pathlib.Path(config['data_root'])
    img_root = data_root / "ADE20K_2021_17_01/images/ADE"
    # This is where we will save the processed data
    proc_root = data_root / "processed" / str(config['version'])
    ex_counter = 0
    for split_dir in tqdm(img_root.iterdir(), total=len(list(img_root.iterdir()))):
        for scene_type_dir in split_dir.iterdir():
            for scene_dir in scene_type_dir.iterdir():
                # get all of the files in scene_dir that end in .jpg
                for image_dir in list(scene_dir.glob("*.jpg")):
                    try:
                        raw_img = np.array(Image.open(image_dir))
                        label_dir = image_dir.parent / image_dir.name.replace(".jpg", "_seg.png")
                        raw_lab = np.array(Image.open(label_dir))
                        img, label = convertFromADE(raw_img, raw_lab)
                        assert img.shape[:2] == label.shape.shape[:2], "Image and Labels should have the same shape."

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

                            if ex_counter > num_examples_to_show:
                                break
                            # Only count examples if showing examples
                            ex_counter += 1

                        if config["save"]:
                            example_name = "_".join(image_dir.name.split("_")[:-1])
                            save_root = proc_root / split_dir.name/ example_name
                            
                            if not save_root.exists():
                                save_root.mkdir(parents=True)

                            img_save_dir = save_root / "image.npy"
                            label_save_dir = save_root / "label.npy"

                            np.save(img_save_dir, img)
                            np.save(label_save_dir, label)
                    except Exception as e:
                        print(f"Error with {image_dir.name}: {e}. Skipping")


def thunderify_ADE20K(
    cfg: Config
):
    # Get the configk
    config = cfg.to_dict()
    # Where the data is 
    data_root = pathlib.Path(config['data_root'])
    img_root = data_root / "ADE20K_2021_17_01/images/ADE"

    dst_dir = pathlib.Path(config["dst_dir"]) / str(config["version"])
    # Append version to our paths
    splits_seed = 42
    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        example_dict = {} 
        # Iterate through the examples.
        for split_dir in tqdm(img_root.iterdir(), total=len(list(img_root.iterdir()))):
            example_dict[split_dir.name] = []
            print(f"Processing {split_dir.name}")
            for scene_type_dir in split_dir.iterdir():
                print(f"Processing {scene_type_dir.name}")
                for scene_dir in scene_type_dir.iterdir():
                    print(f"Processing {scene_dir.name}")
                    # get all of the files in scene_dir that end in .jpg
                    for image_dir in list(scene_dir.glob("*.jpg")):
                        try:
                            # Example name
                            # load the image and label
                            raw_img = np.array(Image.open(image_dir))
                            label_dir = image_dir.parent / image_dir.name.replace(".jpg", "_seg.png")
                            raw_lab = np.array(Image.open(label_dir))
                            img, seg = convertFromADE(raw_img, raw_lab)
                            assert img.shape[:2] == seg.shape[:2], "Image and Labels should have the same shape."
                            # Convert to the right type
                            img = img.astype(np.float32).transpose(2, 0, 1)
                            seg = seg.astype(np.int64)
                            H, W = seg.shape

                            assert img.shape == (3, H, W), f"Image shape isn't correct, got {img.shape}"
                            assert seg.shape == (H, W), f"Seg shape isn't correct, got {seg.shape}"
                            assert np.count_nonzero(seg) > 0, "Label can't be empty."

                            # Save the datapoint to the database
                            example_name = scene_type_dir.name + "_" + scene_dir.name + "_" + image_dir.with_suffix('').name.split("_")[-1]
                            key = example_name

                            db[key] = (img, seg) 
                            example_dict[split_dir.name].append(key)   
                        except Exception as e:
                            print(f"Error with {key}: {e}. Skipping")

        # Split the data into train, cal, val, test
        train_examples = sorted(example_dict["train"])
        valcaltest_examples = sorted(example_dict["val"])

        valcal_examples, test_examples = train_test_split(valcaltest_examples, test_size=0.25, random_state=splits_seed)
        val_examples, cal_examples = train_test_split(valcal_examples, test_size=0.5, random_state=splits_seed)

        # Accumulate the examples
        examples = train_examples + val_examples + cal_examples + test_examples

        # Extract the ids
        data_ids = [ex.split("_")[-1] for ex in examples]
        scene_type = [ex.split("_")[0] for ex in examples]
        scene = ["".join(ex.split("_")[1:-1]) for ex in examples]

        splits = {
            "train": train_examples,
            "val": val_examples,
            "cal": cal_examples,
            "test": test_examples
        }

        # Save the metadata
        db["_examples"] = examples 
        db["_samples"] = examples 
        db["_ids"] = data_ids 
        db["_scene_type"] = scene_type 
        db["_scene"] = scene 
        db["_splits"] = splits
        attrs = dict(
            dataset="CityScapes",
            version=config["version"],
        )
        db["_splits"] = splits
        db["_attrs"] = attrs


def convertFromADE(img_np, lab_np):
    indexMapping = np.loadtxt('/local/vbutoi/projects/misc/research-code/ese/sceneparsing/convertFromADE/mapFromADE.txt').astype(int)
    # Ensure image and label are of the same dimensions
    assert img_np.shape[:2] == lab_np.shape[:2], "Image and label dimensions mismatch!"

    # Resize
    h, w = img_np.shape[:2]
    h_new, w_new = h, w

    if h < w and h > 512:
        h_new = 512
        w_new = int(round(w / h * 512))
    elif w < h and w > 512:
        h_new = int(round(h / w * 512))
        w_new = 512

    img_np = np.array(Image.fromarray(img_np).resize((w_new, h_new), Image.BILINEAR))
    lab_np_resized = np.array(Image.fromarray(lab_np).resize((w_new, h_new), Image.NEAREST))

    # Convert
    labOut_np = convert(lab_np_resized, indexMapping)

    return img_np, labOut_np

def convert(lab, indexMapping):
    # Resize
    h, w = lab.shape[:2]
    h_new, w_new = h, w

    if h < w and h > 512:
        h_new = 512
        w_new = int(round(w / h * 512))
    elif w < h and w > 512:
        h_new = int(round(h / w * 512))
        w_new = 512

    lab = np.array(Image.fromarray(lab).resize((w_new, h_new), Image.NEAREST))

    # Map index
    labADE = (lab[:, :, 0].astype(np.uint16) // 10) * 256 + lab[:, :, 1].astype(np.uint16)
    labOut = np.zeros(labADE.shape, dtype=np.uint8)

    classes_unique = np.unique(labADE)
    for cls in classes_unique:
        if np.sum(cls == indexMapping[:, 1]) > 0:
            labOut[labADE == cls] = indexMapping[cls == indexMapping[:, 1], 0]

    return labOut
