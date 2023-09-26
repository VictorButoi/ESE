# local imports
from ..augmentation import augmentations_from_config

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
        # Build the augmentations if they are there to be built.        
        config_dict = self.config.to_dict()

        if "augmentations" in config_dict and (config_dict["augmentations"] is not None):
            self.aug_pipeline = augmentations_from_config(config_dict["augmentations"])
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
            in_channels = data_config.pop("in_channels")
            out_channels = data_config.pop("out_channels")
            assert out_channels > 1, "Must be multi-class segmentation!"
            model_config["in_channels"] = in_channels
            model_config["out_channels"] = out_channels 

        self.config = Config(total_config)

        self.model = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)
    
    def build_loss(self):
        # If there is a composition of losses, then combine them.
        # else just build the loss normally.
        if "classes" in self.config["loss_func"]: 
            loss_classes = self.config["loss_func"]["classes"]
            weights = torch.Tensor(self.config["loss_func"]["weights"]).to(self.device)
            # Get the loss functions you're using.
            loss_funcs = eval_config(loss_classes)
            # Get the weights per loss and normalize to weights to 1.
            loss_weights = (weights / torch.sum(weights))
            # Build the loss function.
            self.loss_func = lambda yhat, y: torch.sum([loss_weights[l_idx] * l_func(yhat, y) for l_idx, l_func in enumerate(loss_funcs)])
        else:
            self.loss_func = eval_config(self.config["loss_func"])
    
    def run_step(self, batch_idx, batch, backward=True, augmentation=False, epoch=None, phase=None):

        # Send data and labels to device.
        x, y = to_device(batch, self.device)
        
        if ("slice_batch_size" in self.config["data"]) and (self.config["data"]["slice_batch_size"] > 1):
            # This lets you potentially use multiple slices from 3D volumes by mixing them into a big batch.
            img = einops.rearrange(img, "b c h w -> (b c) 1 h w")
            y = einops.rearrange(y, "b c h w -> (b c) 1 h w")

        # Add augmentation to image and label.
        with torch.no_grad():
            x, y = self.aug_pipeline(x, y)

        # Forward pass
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
        
        if loss < 0.5:
            # Run step-wise callbacks if you have them.
            self.run_callbacks("step", batch=forward_batch)
            #raise ValueError("Loss is too low, exiting early.")

        return forward_batch

    def run(self):
        super().run()

    def vis_loss_curves(
        self,
        x='epoch',  
        y='dice_score',
        height=12,
    ):

        # Show a lineplot of the loss curves.
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
