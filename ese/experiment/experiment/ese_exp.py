# torch imports
import torch
from torch.utils.data import DataLoader

# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import
from ionpy.util.torchutils import to_device
from ionpy.util.hash import json_digest
from universeg.experiment.augmentation import augmentations_from_config

class CalibrationExperiment(TrainExperiment):

    def build_augmentations(self):
        if "augmentations" in self.config:
            aug_cfg = self.config.to_dict()["augmentations"]
            self.aug_pipeline = augmentations_from_config(aug_cfg)
            self.properties["aug_digest"] = json_digest(self.config["augmentations"])[
                :8
            ]

    def build_data(self):
        data_cfg = self.config["data"].to_dict()
        dataset_cls = absolute_import(data_cfg.pop("_class"))

        dataset = data_cfg.pop("dataset")

        self.train_dataset = dataset_cls(dataset=dataset, split="train", **data_cfg)
        self.cal_dataset = dataset_cls(dataset=dataset, split="cal", **data_cfg)
        self.val_dataset = dataset_cls(dataset=dataset, split="val", **data_cfg)
    
    def build_dataloader(self):
        assert not (self.config["dataloader"]["batch_size"] > 1 and self.config["data"]["slice_batch_size"] > 1), "No mixing of slice_batch_sz and batch_size."
        dl_cfg = self.config["dataloader"]

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_cfg)
    
    def run_step(self, batch_idx, batch, backward=True, augmentation=False, epoch=None, phase=None):

        # Send data and labels to device.
        x, y = to_device(batch, self.device)
        
        # This lets you potentially use multiple slices from 3D volumes.
        if self.config["dataloader"]["batch_size"] == 1:
            x = x[0][:, None, :, :]
            y = y[0][:, None, :, :]
        
        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline(x, y)

        yhat = self.model(x)
        loss = self.loss_func(yhat, y)

        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        
        forward_batch = {
            "x": x,
            "ytrue": y,
            "ypred": yhat,
            "loss": loss,
            "batch_idx": batch_idx,
        }
        self.run_callbacks("step", batch=forward_batch)
        forward_batch.pop("x")

        return forward_batch

    def run(self):
        super().run()
