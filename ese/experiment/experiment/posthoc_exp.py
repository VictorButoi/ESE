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
        if "data" in self.config:
            pretrained_data_cfg.update(self.config["data"].to_dict())
        total_config["data"] = pretrained_data_cfg
        autosave(total_config, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_config)
        self.data_class = pretrained_data_cfg["_class"]
        # Optionally, load the data.
        if load_data:
            # Get the dataset class and build the transforms
            dataset_cls = absolute_import(pretrained_data_cfg.pop("_class"))
            # Build the augmentation pipeline.
            aug_exists = "augmentations" in total_config and (total_config["augmentations"] is not None)
            if "add_aug" in pretrained_data_cfg:
                add_aug = pretrained_data_cfg.pop("add_aug")
            else:
                add_aug = True
            # If there are augmentations to add and we want to add them, then add them.
            if aug_exists and add_aug:
                val_transforms = augmentations_from_config(total_config["augmentations"]["val"])
                cal_transforms = augmentations_from_config(total_config["augmentations"]["cal"])
                self.properties["aug_digest"] = json_digest(self.config["augmentations"].to_dict())[
                    :8
                ]
            else:
                val_transforms, cal_transforms = None, None
            # Build the datasets, apply the transforms
            train_splits = pretrained_data_cfg.pop("train_splits")
            val_splits = pretrained_data_cfg.pop("val_splits")
            self.train_dataset = dataset_cls(split=train_splits, transforms=val_transforms, **pretrained_data_cfg)
            self.val_dataset = dataset_cls(split=val_splits, transforms=cal_transforms, **pretrained_data_cfg)
            # Check if we want to cache the dataset on the GPU.
            if "cuda" in pretrained_data_cfg:
                assert pretrained_data_cfg["preload"], "If you want to cache the dataset on the GPU, you must preload it."
                cache_dsets_on_gpu = pretrained_data_cfg.pop("cuda")
            else:
                cache_dsets_on_gpu = False

            # Optionally cache the datasets on the GPU.
            if cache_dsets_on_gpu:
                self.train_dataset = CUDACachedDataset(self.train_dataset)
                self.val_dataset = CUDACachedDataset(self.val_dataset)
    
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
        ###################
        # BUILD THE MODEL #
        ###################
        # Get the configs of the experiment
        load_exp_args = {
            "device": "cuda",
            "load_data": False, # Important, we might want to modify the data construction.
        }
        if "checkpoint" in total_config['train']:
            load_exp_args["checkpoint"] = total_config['train']['checkpoint']
        else:
            load_exp_args["checkpoint"] = 'max-val-dice_score' 
            
        if "config.yml" in os.listdir(total_config['train']['pretrained_dir']):
            self.pretrained_exp = load_experiment(
                path=total_config['train']['pretrained_dir'],
                **load_exp_args
            )
        else:
            rs = ResultsLoader()
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(rs.load_configs(total_config['train']['pretrained_dir'], properties=False)),
                selection_metric=total_config['train']['pretrained_select_metric'],
                **load_exp_args
            )
        pretrained_cfg = self.pretrained_exp.config.to_dict()
        #########################################
        #            Model Creation             #
        #########################################
        model_config_dict = total_config['model']
        if '_pretrained_class' in model_config_dict:
            model_config_dict.pop('_pretrained_class')
        self.model_class = model_config_dict['_class']
        # Either keep training the network, or use a post-hoc calibrator.
        if self.model_class in ["Vanilla", "FT_CE", "FT_Dice"]:
            self.base_model = torch.nn.Identity()
            # Load the model, there is no learned calibrator.
            self.model = self.pretrained_exp.model
        else:
            self.base_model = self.pretrained_exp.model
            self.base_model.eval()
            # Get the old in and out channels from the pretrained model.
            model_config_dict["num_classes"] = pretrained_cfg['model']['out_channels']
            model_config_dict["image_channels"] = pretrained_cfg['model']['in_channels']
            # BUILD THE CALIBRATOR #
            ########################
            # Load the model
            self.model = eval_config(model_config_dict)
            self.model.weights_init()
        # Get the tuned calibration parameters.
        self.properties["num_params"] = num_params(self.model)
        ########################################################################
        # Make sure we use the old experiment seed and add important metadata. #
        ########################################################################
        old_exp_config = self.pretrained_exp.config.to_dict() 
        total_config['experiment'] = old_exp_config['experiment']
        model_config_dict['_class'] = self.model_class
        model_config_dict['_pretrained_class'] = parse_class_name(str(self.base_model.__class__))
        autosave(total_config, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_config)
    
    def build_loss(self):
        total_config = self.config.to_dict()
        if self.model_class in ["FT_CE", "FT_Dice"]:
            # Load the config for the loss function.
            loss_cfg_root = Path("/storage/vbutoi/projects/ESE/ese/experiment/configs/losses")
            ##################################################
            dataset_name = self.data_class.split(".")[-1]
            with open(loss_cfg_root / f"{dataset_name}.yaml", 'r') as file:
                loss_cfg = yaml.safe_load(file)
            if self.model_class == "FT_CE":
                new_loss_cfg = loss_cfg["ce_loss"]
            else:
                new_loss_cfg = loss_cfg["dice_loss"]
            ##################################################
            total_config["loss_func"] = new_loss_cfg
            autosave(total_config, self.path / "config.yml") # Save the new config because we edited it.
            self.config = Config(total_config)
        # Build the loss function.
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
        if self.model_class in ["Vanilla", "FT_CE", "FT_Dice"]:
            yhat_cal = self.model(yhat)
        else:
            yhat_cal = self.model(yhat, image=x)
        # Calculate the loss between the pred and original preds.
        if "BWPixelCELoss" in str(self.loss_func):
            loss = self.loss_func(yhat_cal, y, dist_to_boundary=batch["dist_to_boundary"])
        else:
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
        assert x.shape[0] == 1, "Batch size must be 1 for prediction for now."

        # Predict with the base model.
        with torch.no_grad():
            yhat = self.base_model(x)

        # Apply post-hoc calibration.
        if self.model_class in ["Vanilla", "FT_CE", "FT_Dice"]:
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
