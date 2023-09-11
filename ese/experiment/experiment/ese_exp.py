# local imports
from ese.experiment.augmentation import build_transforms

# torch imports
import torch
from torch.utils.data import DataLoader

# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.nn.util import num_params
from ionpy.util import Config
from ionpy.util.hash import json_digest
from ionpy.util.torchutils import to_device

# misc imports
import einops
import seaborn as sns


class CalibrationExperiment(TrainExperiment):

    def build_augmentations(self):
        if self.config["train"]["augmentations"] != "None":
            raise NotImplementedError("Augmentations not implemented for calibration.")

            aug_cfg = self.config.to_dict()["augmentations"]
            self.properties["aug_digest"] = json_digest(self.config["augmentations"])[
                :8
            ]

    def build_data(self):
        # Get the data and transforms we want to apply
        data_cfg = self.config["data"].to_dict()

        # Get the dataset class and build the transforms
        dataset_cls = absolute_import(data_cfg.pop("_class"))
        
        # Build the datasets, apply the transforms
        self.train_dataset = dataset_cls(split="train", **data_cfg)
        self.cal_dataset = dataset_cls(split="cal", **data_cfg)
        self.val_dataset = dataset_cls(split="val", **data_cfg)
    
    def build_dataloader(self):
        dl_cfg = self.config["dataloader"]
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_cfg)

    def build_model(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()

        # Get the model and data configs.
        model_config = total_config["model"]
        data_config = total_config["data"]

        # transfer the arguments to the model config.
        if "in_channels" in data_config:
            model_config["in_channels"] = data_config.pop("in_channels")
            model_config["out_channels"] = data_config.pop("out_channels")

        self.config = Config(total_config)

        self.model = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)
    
    def run_step(self, batch_idx, batch, backward=True, augmentation=False, epoch=None, phase=None):

        # Send data and labels to device.
        x, y = to_device(batch, self.device)
        
        if ("slice_batch_size" in self.config["data"]) and (self.config["data"]["slice_batch_size"] > 1):
            # This lets you potentially use multiple slices from 3D volumes by mixing them into a big batch.
            img = einops.rearrange(img, "b c h w -> (b c) 1 h w")
            mask = einops.rearrange(mask, "b c h w -> (b c) 1 h w")

        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline(x, y)

        # Forward pass
        yhat = self.model(x)
        
        # Get the loss (segmentation loss)
        loss = self.loss_func(yhat, y)

        print(loss)

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
