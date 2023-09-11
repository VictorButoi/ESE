import os
from typing import Literal, Any
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset

# ionpy imports
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class PascalVOC(Dataset):

    split: Literal["train", "cal", "val"] = "train"
    transform: Any = None

    def __post_init__(self):
        root = "/storage/vbutoi/datasets/VOCdevkit/VOC2012"
        self.files = list(map(lambda x: x.strip(), open(os.path.join(root, 'ImageSets', 'Segmentation', self.split + '.txt')).readlines()))
        if self.is_transform:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            self.label_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = self.files[index]
        image_path = os.path.join(self.root, 'JPEGImages', image_name + '.jpg')
        label_path = os.path.join(self.root, 'SegmentationClass', image_name + '.png')
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        
        if self.is_transform:
            image = self.transform(image)
            label = self.label_transform(label)
            label = np.array(label)  # Convert to numpy array
            label = torch.from_numpy(label).long()  # Convert to tensor
        
        return image, label