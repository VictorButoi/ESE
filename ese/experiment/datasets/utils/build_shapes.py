import numpy as np
import pathlib
import numpy as np
from thunderpack import ThunderDB
from tqdm import tqdm
from ionpy.util import Config

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from ionpy.experiment.util import fix_seed
import neurite_sandbox as nes
from pydantic import validate_arguments


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


def thunderify_Shapes(
    cfg: Config
):
    config = cfg.to_dict()
    # Append version to our paths
    dst_dir = pathlib.Path(config["dst_dir"]) / str(config["version"])

    # Append version to our paths
    splits_seed = 42

    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        example_dict = {} 
        # Iterate through the examples.
        for split_dir in proc_root.iterdir():
            print("Doing split", split_dir.name)
            example_dict[split_dir.name] = []
            for example_dir in tqdm(split_dir.iterdir(), total=len(list(split_dir.iterdir()))):
                # Example name
                key = example_dir.name
                # Paths to the image and segmentation
                img_dir = example_dir / "image.npy"
                seg_dir = example_dir / "label.npy"
                try:
                    # Load the image and segmentation.
                    img = np.load(img_dir)
                    img = img.transpose(2, 0, 1)
                    seg = np.load(seg_dir)
                    
                    # Convert to the right type
                    img = img.astype(np.float32)
                    seg = seg.astype(np.int64)
                    
                    # Save the datapoint to the database
                    db[key] = (img, seg) 
                    example_dict[split_dir.name].append(key)   
                except Exception as e:
                    print(f"Error with {key}: {e}. Skipping")

        # Split the data into train, cal, val, test
        train_examples = sorted(example_dict["train"])
        valcal_examples = sorted(example_dict["val"])
        val_examples, cal_examples = train_test_split(valcal_examples, test_size=0.5, random_state=splits_seed)
        test_examples = sorted(example_dict["test"])

        # Accumulate the examples
        examples = train_examples + val_examples + cal_examples + test_examples

        # Extract the ids
        data_ids = ["_".join(ex.split("_")[1:]) for ex in examples]
        cities = [ex.split("_")[0] for ex in examples]

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
        db["_cities"] = cities 
        db["_splits"] = splits
        attrs = dict(
            dataset="CityScapes",
            version=config["version"],
        )
        db["_splits"] = splits
        db["_attrs"] = attrs


def perlin_generation(
    synth_cfg: dict
):
    gen_opts_cfg = synth_cfg['gen_opts']
    aug_cfg = synth_cfg['augmentations']

    fix_seed(gen_opts_cfg["seed"])

    # Gen parameters
    if gen_opts_cfg['num_labels_range'][0] == gen_opts_cfg['num_labels_range'][1]:
        num_labels = gen_opts_cfg['num_labels_range'][0]
    else:
        num_labels = np.random.randint(low=gen_opts_cfg['num_labels_range'][0], high=gen_opts_cfg['num_labels_range'][1])

    # Set the augmentation parameters.
    if aug_cfg['shapes_im_max_std_range'][0] == aug_cfg['shapes_im_max_std_range'][1]:
        shapes_im_max_std = aug_cfg['shapes_im_max_std_range'][0]
    else:
        shapes_im_max_std = np.random.uniform(aug_cfg['shapes_im_max_std_range'][0], aug_cfg['shapes_im_max_std_range'][1])
    
    if aug_cfg['shapes_warp_max_std_range'][0] == aug_cfg['shapes_warp_max_std_range'][1]:
        shapes_warp_max_std = aug_cfg['shapes_warp_max_std_range'][0]
    else:
        shapes_warp_max_std = np.random.uniform(aug_cfg['shapes_warp_max_std_range'][0], aug_cfg['shapes_warp_max_std_range'][1])
    
    if aug_cfg['std_min_range'][0] == aug_cfg['std_min_range'][1]:
        std_min = aug_cfg['std_min_range'][0]
    else:
        std_min = np.random.uniform(aug_cfg['std_min_range'][0], aug_cfg['std_min_range'][1])
        
    if aug_cfg['std_max_range'][0] == aug_cfg['std_max_range'][1]:
        std_max = aug_cfg['std_max_range'][0]
    else:
        std_max = np.random.uniform(aug_cfg['std_max_range'][0], aug_cfg['std_max_range'][1])

    if aug_cfg['lab_int_interimage_std_range'][0] == aug_cfg['lab_int_interimage_std_range'][1]:
        lab_int_interimage_std = aug_cfg['lab_int_interimage_std_range'][0]
    else:
        lab_int_interimage_std = np.random.uniform(aug_cfg['lab_int_interimage_std_range'][0], aug_cfg['lab_int_interimage_std_range'][1])

    if aug_cfg['warp_std_range'][0] == aug_cfg['warp_std_range'][1]:
        warp_std = aug_cfg['warp_std_range'][0]
    else:
        warp_std = np.random.uniform(aug_cfg['warp_std_range'][0], aug_cfg['warp_std_range'][1])

    if aug_cfg['bias_res_range'][0] == aug_cfg['bias_res_range'][1]:
        bias_res = aug_cfg['bias_res_range'][0]
    else:
        bias_res = np.random.uniform(aug_cfg['bias_res_range'][0], aug_cfg['bias_res_range'][1])

    if aug_cfg['bias_std_range'][0] == aug_cfg['bias_std_range'][1]:
        bias_std = aug_cfg['bias_std_range'][0]
    else:
        bias_std = np.random.uniform(aug_cfg['bias_std_range'][0], aug_cfg['bias_std_range'][1])

    if aug_cfg['blur_std_range'][0] == aug_cfg['blur_std_range'][1]:
        blur_std = aug_cfg['blur_std_range'][0]
    else:
        blur_std = np.random.uniform(aug_cfg['blur_std_range'][0], aug_cfg['blur_std_range'][1])

    # Gen tasks
    images, label_maps, _ = nes.tf.utils.synth.perlin_nshot_task(in_shape=gen_opts_cfg['img_res'],
                                                                  num_gen=gen_opts_cfg['num_to_gen'],
                                                                  num_label=num_labels,
                                                                  shapes_im_scales=gen_opts_cfg['shapes_im_scales'],
                                                                  shapes_warp_scales=gen_opts_cfg['shapes_warp_scales'],
                                                                  shapes_im_max_std=shapes_im_max_std,
                                                                  shapes_warp_max_std=shapes_warp_max_std,
                                                                  min_int=0,
                                                                  max_int=1,
                                                                  std_min=std_min,
                                                                  std_max=std_max,
                                                                  lab_int_interimage_std=lab_int_interimage_std,
                                                                  warp_std=warp_std,
                                                                  warp_res=gen_opts_cfg['warp_res'],
                                                                  bias_res=bias_res,
                                                                  bias_std=bias_std,
                                                                  blur_std=blur_std)
    
    return images, label_maps, _ 