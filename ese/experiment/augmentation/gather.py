import ast
import torch
import torchvision.transforms as transforms


def augmentations_from_config(aug_config_dict_list):
    # Get the transforms corresponding to training.
    return transforms.Compose([
        build_aug(aug_cfg) for aug_cfg in aug_config_dict_list
    ])


def build_aug(augmentation_dict):
    aug_key = list(augmentation_dict.keys())[0]
    aug_cfg = augmentation_dict[aug_key]
    # List of all possible augmentations
    if aug_key == "ColorJitter":
        return transforms.ColorJitter(**augmentation_dict[aug_key])
    elif aug_key == "RandomResizedCrop":
        return transforms.RandomResizedCrop(size=ast.literal_eval(aug_cfg["size"]))
    elif aug_key == "RandomHorizontalFlip":
        return transforms.RandomHorizontalFlip(**augmentation_dict[aug_key])
    elif aug_key == "Resize":
        return transforms.Resize(size=ast.literal_eval(aug_cfg["size"]))
    elif aug_key == "Normalize":
        mean = ast.literal_eval(aug_cfg["mean"])
        std = ast.literal_eval(aug_cfg["std"])
        return transforms.Normalize(mean=mean, std=std)
    else:
        raise ValueError("Unknown augmentation: {}".format(aug_key))