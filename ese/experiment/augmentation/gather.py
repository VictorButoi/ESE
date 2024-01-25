import ast
import albumentations as A


def augmentations_from_config(aug_config_dict_list):
    # Get the A corresponding to training.
    return A.Compose([
        build_aug(aug_cfg) for aug_cfg in aug_config_dict_list
    ])


def build_aug(augmentation_dict):
    aug_key = list(augmentation_dict.keys())[0]
    aug_cfg = augmentation_dict[aug_key]
    # List of all possible augmentations
    if aug_key == "ColorJitter":
        return A.ColorJitter(**augmentation_dict[aug_key])
    elif aug_key == "RandomResizedCrop":
        return A.RandomResizedCrop(size=ast.literal_eval(aug_cfg["size"]))
    elif aug_key == "RandomHorizontalFlip":
        return A.RandomHorizontalFlip(**augmentation_dict[aug_key])
    elif aug_key == "Resize":
        return A.Resize(size=ast.literal_eval(aug_cfg["size"]))
    elif aug_key == "Normalize":
        mean = ast.literal_eval(aug_cfg["mean"])
        std = ast.literal_eval(aug_cfg["std"])
        return A.Normalize(mean=mean, std=std)
    else:
        raise ValueError("Unknown augmentation: {}".format(aug_key))