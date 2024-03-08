import albumentations
from ionpy.experiment.util import absolute_import


def augmentations_from_config(aug_config_dict_list):
    return albumentations.Compose([
        build_aug(aug_cfg) for aug_cfg in aug_config_dict_list
    ])

def build_aug(aug_obj):
    if isinstance(aug_obj, dict):
        aug_key = list(aug_obj.keys())[0]
        return absolute_import(f'albumentations.{aug_key}')(**aug_obj[aug_key])
    else:
        return absolute_import(f'albumentations.{aug_obj}')()
