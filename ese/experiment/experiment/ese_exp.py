# misc imports
import pathlib

# torch imports
import torch

# diseg imports
from diseg.experiment.datasets import ModifiedCUDACachedDataset
from diseg.experiment.augmentation import augmentations_from_config

# IonPy imports
from IonPy.experiment import TrainExperiment
from IonPy.experiment.util import absolute_import, eval_config
from IonPy.nn.util import num_params
from IonPy.util.hash import json_digest


class ESExperimet(TrainExperiment):

    def build_data(self):
        
    def build_augmentations(self):

    def build_model(self):
    
    def build_loss(self):

    def run_step(self, *, batch_idx, batch, backward=True, augmentation=None, epoch=None, **kwargs):

    def run(self):
        super().run()
