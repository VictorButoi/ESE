# Torch imports
from torch.utils.data import Dataset

# Local imports
from .utils.ade_utils import loadAde20K_file

# IonPy imports
from ionpy.util.validation import validate_arguments_init

# Validation Imports
from dataclasses import dataclass
from ESE.ese.experiment.datasets.wmh import Segment2D

# Misc imports
import os
import pickle
from PIL import Image

# Dataset for the Ade20K dataset
@validate_arguments_init
@dataclass
class ADE20kDataset(Dataset):

    def __post_init__(self):
        # Call the constructor for PyTorch dataset
        super().__init__()

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


