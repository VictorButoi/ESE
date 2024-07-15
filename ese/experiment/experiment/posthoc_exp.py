# local imports
from ..augmentation.gather import augmentations_from_config
from .utils import load_experiment, process_pred_map, parse_class_name
# torch imports
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
# IonPy imports
from ionpy.util import Config
from ionpy.nn.util import num_params
from ionpy.util.ioutil import autosave
from ionpy.util.hash import json_digest
from ionpy.analysis import ResultsLoader
from ionpy.util.torchutils import to_device
from ionpy.experiment import TrainExperiment
from ionpy.datasets.cuda import CUDACachedDataset
from ionpy.experiment.util import absolute_import, eval_config
# misc imports
import os
from typing import Optional


class PostHocExperiment(TrainExperiment):

    def build_data(self, load_data):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        # Get the data and transforms we want to apply
        pretrained_data_cfg = self.pretrained_exp.config["data"].to_dict()
        # Update the old cfg with new cfg (if it exists).
        pretrained_data_cfg.update(total_config.get("data", {}))
        new_data_cfg = pretrained_data_cfg.copy()
        # Finally update the data config with the copy. 
        total_config["data"] = new_data_cfg 

         # Save the new config because we edited it.
        autosave(total_config, self.path / "config.yml")
        self.config = Config(total_config)

        # Get the dataset class and build the transforms
        dataset_cls = absolute_import(new_data_cfg.pop("_class"))
        # Build the augmentation pipeline.
        augmentation_list = total_config.get("augmentations", None)
        if augmentation_list is not None:
            train_transforms = augmentations_from_config(augmentation_list.get("train", None))
            val_transforms = augmentations_from_config(augmentation_list.get("val", None))
            self.properties["aug_digest"] = json_digest(augmentation_list)[:8]
        else:
            train_transforms, val_transforms = None, None

        if load_data:
            if "train_splits" in new_data_cfg and "val_splits" in new_data_cfg:
                train_splits = new_data_cfg.pop("train_splits")
                val_splits = new_data_cfg.pop("val_splits")
                self.train_dataset = dataset_cls(split=train_splits, transforms=train_transforms, **new_data_cfg)
                self.val_dataset = dataset_cls(split=val_splits, transforms=val_transforms, **new_data_cfg)
            else:
                self.train_dataset = dataset_cls(split="train", transforms=train_transforms, **new_data_cfg)
                self.val_dataset = dataset_cls(split="val", transforms=val_transforms, **new_data_cfg)

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
        total_cfg_dict = self.config.to_dict()

        #######################
        # LOAD THE EXPERIMENT #
        #######################
        # Get the configs of the experiment
        load_exp_args = {
            "device": "cuda",
            "load_data": False, # Important, we might want to modify the data construction.
            "checkpoint": total_cfg_dict['train'].get('checkpoint', 'max-val-dice_score')
        }
            
        # Either select from a set of experiments in a common directory OR choose a particular experiment to load.
        if "config.yml" in os.listdir(total_cfg_dict['train']['pretrained_dir']):
            self.pretrained_exp = load_experiment(
                path=total_cfg_dict['train']['pretrained_dir'],
                **load_exp_args
            )
        else:
            rs = ResultsLoader()
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(rs.load_configs(total_cfg_dict['train']['pretrained_dir'], properties=False)),
                selection_metric=total_cfg_dict['train']['pretrained_select_metric'],
                **load_exp_args
            )
        # Now we can access the old total config. 
        pretrained_total_cfg_dict = self.pretrained_exp.config.to_dict()

        #########################################
        #            Model Creation             #
        #########################################
        model_cfg_dict = total_cfg_dict['model']
        pretrained_model_cfg_dict = pretrained_total_cfg_dict['model']
        # Either keep training the network, or use a post-hoc calibrator.
        self.model_class = model_cfg_dict['_class']
        if self.model_class is None:
            self.base_model = torch.nn.Identity()
            # Load the model, there is no learned calibrator.
            self.model = self.pretrained_exp.model
            # Edit the model_config.
            total_cfg_dict['model']['_class'] = parse_class_name(str(self.base_model.__class__))
            total_cfg_dict['model']['_pretrained_class'] = parse_class_name(str(self.model.__class__))
        else:
            self.base_model = self.pretrained_exp.model
            self.base_model.eval()
            # Get the old in and out channels from the pretrained model.
            model_cfg_dict["num_classes"] = pretrained_model_cfg_dict['out_channels']
            model_cfg_dict["image_channels"] = pretrained_model_cfg_dict['in_channels']
            # BUILD THE CALIBRATOR #
            ########################
            # Load the model
            self.model = eval_config(model_cfg_dict)
            self.model.weights_init()
            # Edit the model_config, note that this is flipped with above.
            total_cfg_dict['model']['_class'] = parse_class_name(str(self.model.__class__))
            total_cfg_dict['model']['_pretrained_class'] = parse_class_name(str(self.base_model.__class__))


        ########################################################################
        # Make sure we use the old experiment seed and add important metadata. #
        ########################################################################
        # Get the tuned calibration parameters.
        self.properties["num_params"] = num_params(self.model)
        # Set the new experiment params as the old ones.
        total_cfg_dict['experiment'] = pretrained_total_cfg_dict['experiment']
        # Save the new config because we edited it and reset self.config
        autosave(total_cfg_dict, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_cfg_dict)
    
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
            self.loss_func = lambda yhat, y: sum([loss_weights[l_idx] * l_func(yhat, y) for l_idx, l_func in enumerate(loss_funcs)])
        else:
            self.loss_func = eval_config(self.config["loss_func"])
    
    def run_step(self, batch_idx, batch, backward, **kwargs):
        # Send data and labels to device.
        batch = to_device(batch, self.device)
        # Get the image and label from the batch.
        x = batch["img"]
        y = batch["label"]
        # Forward pass
        with torch.no_grad():
            yhat = self.base_model(x)
        # Calibrate the predictions.
        if self.model_class is None:
            yhat_cal = self.model(yhat)
        else:
            yhat_cal = self.model(yhat, image=x)
        # Calculate the loss between the pred and original preds.
        loss = self.loss_func(yhat_cal, y)
        # If backward then backprop the gradients.
        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        # Run step-wise callbacks if you have them.
        forward_batch = {
            "x": x,
            "y_true": y,
            "y_pred": yhat_cal,
            "y_logits": yhat_cal, # Used for visualization functions.
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
        # Predict with the base model.
        with torch.no_grad():
            yhat = self.base_model(x)

        # Apply post-hoc calibration.
        if self.model_class is None:
            logit_map = self.model(yhat)
        else:
            logit_map = self.model(yhat, image=x)

        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            logit_map, 
            multi_class=multi_class, 
            threshold=threshold,
            from_logits=True
        )

        # If label is not None, then only return the predictions for that label.
        if label is not None:
            logit_map = logit_map[:, label, ...].unsqueeze(1)
            prob_map = prob_map[:, label, ...].unsqueeze(1)

        # Return the outputs
        return {
            'y_logits': logit_map,
            'y_probs': prob_map, 
            'y_hard': pred_map 
        }

    def to_device(self):
        self.base_model = to_device(self.base_model, self.device, channels_last=False)
        self.model = to_device(self.model, self.device, channels_last=False)
