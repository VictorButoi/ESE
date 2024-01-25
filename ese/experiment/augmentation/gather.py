import torch
from torchvision.transforms import v2


def augmentations_from_config(aug_config):
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    val_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    return train_transforms, val_transforms
