import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Dataset for the Ade20K dataset
class Ade20KDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, train=True, test=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            train (bool): If True, load the training set. If False, load the
                test set.
            test (bool): If True, load the test set. If False, load the
                training set.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.test = test

        # Load the filenames
        self.filenames = []
        if self.train:
            self.filenames = open(os.path.join(self.root_dir, 'train.txt')).read().splitlines()
        elif self.test:
            self.filenames = open(os.path.join(self.root_dir, 'test.txt')).read().splitlines()
        else:
            self.filenames = open(os.path.join(self.root_dir, 'val.txt')).read().splitlines()

        # Load the labels
        self.labels = []
        if self.train:
            self.labels = open(os.path.join(self.root_dir, 'train_labels.txt')).read().splitlines()
        elif self.test:
            self.labels = open(os.path.join(self.root_dir, 'test_labels.txt')).read().splitlines()
        else:
            self.labels = open(os.path.join(self.root_dir, 'val_labels.txt')).read().splitlines()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load the image
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255

        # Load the label
        label_name = os.path.join(self.root_dir, self.labels[idx])
        label = Image.open(label_name)
        label = np.array(label)
        label = np.transpose(label, (2, 0, 1))
        label = label.astype(np.float32)
        label = label / 255

        # Apply the transforms
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

