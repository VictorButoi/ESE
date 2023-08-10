import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
from .ade_utils import loadAde20K_file

# Dataset for the Ade20K dataset
class ADE20kDataset(Dataset):
    
    def __init__(self, split, transform=None):
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
        self.root_dir = "/storage/vbutoi/datasets/ade20k"
        pickl_file = self.root_dir + "/ADE20K_2021_17_01/index_ade20k.pkl"

        # Load data from the pickle file
        with open(pickl_file, 'rb') as pickle_file:
            self.loaded_data = pickle.load(pickle_file)
        
        # Get the filenames
        folders = self.loaded_data['folder']
        filenames = self.loaded_data['filename']
        datapoints = [os.path.join(self.root_dir, folder, filename) for folder, filename in zip(folders, filenames)]

        # Load the images and process the labels
        self.data = datapoints 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        labels = loadAde20K_file(self.data[idx])

        return image, labels 

