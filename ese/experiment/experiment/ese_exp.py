# torch imports
import torch
from torch.utils.data import DataLoader

# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import
from ionpy.util.torchutils import to_device
from ionpy.util.hash import json_digest
from universeg.experiment.augmentation import augmentations_from_config

# misc imports
import einops
import seaborn as sns


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
        self.train_dataset = dataset_cls(split="train", **data_cfg)
        self.cal_dataset = dataset_cls(split="cal", **data_cfg)
        self.val_dataset = dataset_cls(split="val", **data_cfg)
    
    def build_dataloader(self):
        dl_cfg = self.config["dataloader"]
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_cfg)
    
    def run_step(self, batch_idx, batch, backward=True, augmentation=False, epoch=None, phase=None):

        # Send data and labels to device.
        x, y = to_device(batch, self.device)
        
        if self.config["data"]["slice_batch_size"] > 1:
            # This lets you potentially use multiple slices from 3D volumes by mixing them into a big batch.
            img = einops.rearrange(img, "b c h w -> (b c) 1 h w")
            mask = einops.rearrange(mask, "b c h w -> (b c) 1 h w")

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

    def vis_loss_curves(
        self,
        x='epoch',  
        y='dice_score',
        height=12,
    ):

        g = sns.relplot(
            data=self.logs,
            x=x,
            y=y,
            col='phase',
            kind='line',
            height=height,
            )
        # Set column spacing
        g.fig.subplots_adjust(wspace=0.05)
        g.set(ylim=(0, 1))
