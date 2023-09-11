import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class PascalVOC(Dataset):
    def __init__(self, root, is_transform=True, split="train"):
        self.root = root
        self.split = split
        self.files = list(map(lambda x: x.strip(), open(os.path.join(root, 'ImageSets', 'Segmentation', split + '.txt')).readlines()))
        self.is_transform = is_transform
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