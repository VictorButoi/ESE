# misc imports
import pathlib
from threading import Timer

# torch imports
import torch
from torch.utils.data import DataLoader

# IonPy imports
from IonPy.experiment import TrainExperiment
from IonPy.util.torchutils import to_device
from IonPy.experiment.util import absolute_import, eval_config
from IonPy.nn.util import num_params
from IonPy.util.hash import json_digest


class CalibrationExperiment(TrainExperiment):

    def build_data(self):
        dl_cfg = self.config["dataloader"]

        self.train_dataset = self.dataset_class()
        self.val_id_dataset = self.dataset_class(split="val")

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_id_dl = DataLoader(self.val_id_dataset, shuffle=False, drop_last=False, **dl_cfg)
    
    #def build_augmentations(self):

    def build_model(self):
        self.model = self.model_class(**self.config["model"])

    def build_loss(self):
        super().build_loss()

    def run_step(self, batch_idx, batch, backward=True, augmentation=False, epoch=None):

        x, y = batch

        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline(x, y)

        yhat = self.model(x)
        loss = self.loss_func(yhat, y)
        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return {
            "loss": loss,
            "ytrue": y,
            "ypred": yhat,
            "batch_idx": batch_idx,
        }

    def run(self):
        super().run()
