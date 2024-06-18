# local imports
from ionpy.util.meter import MeterDict
from .utils import process_pred_map
from ..augmentation.gather import augmentations_from_config
# torch imports
import torch
from torch.utils.data import DataLoader
# IonPy imports
from ionpy.util import Config
from ionpy.nn.util import num_params
from ionpy.util.hash import json_digest
from ionpy.util.torchutils import to_device
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import, eval_config
# misc imports
import time
import einops
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Literal, Optional


class CalibrationExperiment(TrainExperiment):

    def build_data(self, load_data):
        # Get the data and transforms we want to apply
        total_config = self.config.to_dict()
        data_cfg = total_config["data"]
        # Get the dataset class and build the transforms
        dataset_cls = absolute_import(data_cfg.pop("_class"))
        # Build the augmentation pipeline.
        if "augmentations" in total_config and (total_config["augmentations"] is not None):
            train_transforms = augmentations_from_config(total_config["augmentations"].get("train", None))
            val_transforms = augmentations_from_config(total_config["augmentations"].get("val", None))
            self.properties["aug_digest"] = json_digest(self.config["augmentations"].to_dict())[
                :8
            ]
        else:
            train_transforms, val_transforms = None, None
        # Build the datasets, apply the transforms
        if load_data:
            if "train_splits" in data_cfg and "val_splits" in data_cfg:
                train_splits = data_cfg.pop("train_splits")
                val_splits = data_cfg.pop("val_splits")
                self.train_dataset = dataset_cls(split=train_splits, transforms=train_transforms, **data_cfg)
                self.val_dataset = dataset_cls(split=val_splits, transforms=val_transforms, **data_cfg)
            else:
                self.train_dataset = dataset_cls(split="train", transforms=train_transforms, **data_cfg)
                self.val_dataset = dataset_cls(split="val", transforms=val_transforms, **data_cfg)
    
    def build_dataloader(self, batch_size=None):
        # If the datasets aren't built, build them
        if not hasattr(self, "train_dataset"):
            self.build_data()
        dl_cfg = self.config["dataloader"].to_dict()

        # Optionally manually set the batch size.
        if batch_size is not None:
            dl_cfg["batch_size"] =  batch_size

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
        # Set important things about the model.
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
    
    def run_step(self, batch_idx, batch, backward, **kwargs):
        # Send data and labels to device.
        batch = to_device(batch, self.device)
        # Get the image and label.
        x, y = batch["img"], batch["label"]

        # For volume datasets, sometimes want to treat different slices as a batch.
        if self.config["data"].get("num_slices", 1) != 1:
            # This lets you potentially use multiple slices from 3D volumes by mixing them into a big batch.
            x = einops.rearrange(x, "b c h w -> (b c) 1 h w")
            y = einops.rearrange(y, "b c h w -> (b c) 1 h w")
        
        yhat = self.model(x)

        if yhat.shape[1] > 1:
            y = y.long()

        # Let's visualize the predictions and the ground truth.
        loss = self.loss_func(yhat, y)

        # If backward then backprop the gradients.
        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        # Run step-wise callbacks if you have them.
        forward_batch = {
            "x": x,
            "y_true": y,
            "y_pred": yhat,
            "y_logits": yhat, # Used for visualization functions.
            "loss": loss,
            "batch_idx": batch_idx,
        }
        self.run_callbacks("step", batch=forward_batch)
        return forward_batch

    def predict(
        self, 
        x, 
        multi_class,
        threshold = 0.5,
        label: Optional[int] = None,
    ):
        assert x.shape[0] == 1, "Batch size must be 1 for prediction for now."

        # Get the label predictions
        logit_map = self.model(x) 

        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            logit_map, 
            multi_class=multi_class, 
            threshold=threshold,
            from_logits=True,
        )

        if label is not None:
            logit_map = logit_map[:, label, ...].unsqueeze(1)
            prob_map = prob_map[:, label, ...].unsqueeze(1)

        # Return the outputs
        return {
            'y_logits': logit_map,
            'y_probs': prob_map, 
            'y_hard': pred_map 
        }

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
    
    def vis_predictions(
        self,
        phase: Literal['train', 'val', 'cal'],
        seg_type: Literal['binary', 'multi-class'],
        num_examples: int = 5,
        width: int = 4,
        height: int = 4,
        ):
        # Get the dataloader.
        if not hasattr(self, "train_dl"):
            dl_cfg = self.config["dataloader"].to_dict()
            dl_cfg['batch_size'] = 1 
            self.build_dataloader(dl_cfg)
        dataloader = self.val_dl if phase == 'val' else self.train_dl
        # Set the model to eval mode.
        self.model.eval()
        # Get the examples.
        examples = []
        for idx, batch in enumerate(dataloader):
            # Get the predictions
            with torch.no_grad():
                # Get an image and label and predict for it.
                x, y = to_device(batch, self.device)
                pred_map = self.predict(x)
                # Prepare the data for plotting.
                x = x.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                # If x is rgb
                if x.shape[-1] == 3:
                    x = x.astype(np.uint8)
                    img_cm = None
                else:
                    img_cm = "gray"
                # Prepare the label and prediction for plotting. 
                y = y.squeeze().cpu().numpy()
                pred_map = pred_map.squeeze().cpu().numpy()
                # Add the example to the list.
                examples.append((x, y, pred_map))
            if idx >= num_examples - 1:
                break
        # Get the number of classes.
        if seg_type == "binary":
            num_pred_classes = 2
        else:
            num_pred_classes = self.config['model']['out_channels']
        # Generate a list of random colors, starting with black for background
        if num_pred_classes == 2:
            colors = [(0, 0, 0), (1, 1, 1)]
        else:
            colors = [(0, 0, 0)] + [(np.random.random(), np.random.random(), np.random.random()) for _ in range(num_pred_classes - 1)]
        # Define the colormap
        cmap_name = "seg_map"
        label_cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=num_pred_classes)
        # Visualize the examples in 3 rows (imags, labels, preds).
        f, ax = plt.subplots(num_examples, 3, figsize=(width * height, num_examples * height))
        for idx, (x, y, pred_map) in enumerate(examples):
            # image
            ax[idx, 0].imshow(x, cmap=img_cm, interpolation='None')
            ax[idx, 0].set_title(f"Example {idx}")
            # label
            ax[idx, 1].imshow(y, cmap=label_cm, interpolation='None')
            ax[idx, 1].set_title(f"Label {idx}")
            # prediction
            ax[idx, 2].imshow(pred_map, cmap=label_cm, interpolation='None')
            ax[idx, 2].set_title(f"Prediction {idx}")
            # Set the axes off.
            ax[idx, 0].axis('off')
            ax[idx, 1].axis('off')
            ax[idx, 2].axis('off')
