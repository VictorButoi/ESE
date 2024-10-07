# local imports
from ..augmentation.pipeline import build_aug_pipeline
from .utils import (
    list2tuple,
    load_experiment, 
    process_pred_map, 
    parse_class_name, 
    load_exp_dataset_objs
)
# torch imports
import torch
import torch._dynamo # For compile
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
torch._dynamo.config.suppress_errors = True
# IonPy imports
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.hash import json_digest
from ionpy.analysis import ResultsLoader
from ionpy.util.config import HDict, valmap
from ionpy.util.torchutils import to_device
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.nn.util import num_params, split_param_groups_by_weight_decay
# misc imports
import os
import time
import voxynth
from pprint import pprint
from typing import Optional
import matplotlib.pyplot as plt


class PostHocExperiment(TrainExperiment):

    def build_augmentations(self, load_aug_pipeline):
        super().build_augmentations()
        if "augmentations" in self.config and load_aug_pipeline:
            self.aug_pipeline = build_aug_pipeline(self.config.to_dict()["augmentations"])

    def build_data(self, load_data):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        posthoc_data_cfg = total_config.get("data", {})

        # Get the data and transforms we want to apply
        base_data_cfg = self.pretrained_exp.config["data"].to_dict()

        # Update the old cfg with new cfg (if it exists) and make a copy.
        base_data_cfg.update(posthoc_data_cfg)
        data_cfg = base_data_cfg.copy()

        # If we are using temps as targets, need to add where they are. 
        if (data_cfg['target'] == 'temp') and ('opt_temps_dir' not in data_cfg):
            data_cfg['opt_temps_dir'] = f'{self.pt_model_path}/opt_temps.json'

        # Finally update the data config with the copy. 
        total_config["data"] = data_cfg 
        autosave(total_config, self.path / "config.yml")
        self.config = Config(total_config)

        if load_data:
            # Load the datasets.
            dset_objs = load_exp_dataset_objs(data_cfg, self.properties) 
            # Initialize the dataset classes.
            self.train_dataset = dset_objs['train']
            self.val_dataset = dset_objs['val']

    def build_dataloader(self, batch_size=None):
        # If the datasets aren't built, build them
        if not hasattr(self, "train_dataset"):
            self.build_data()
        dl_cfg = self.config["dataloader"].to_dict()

        # Optionally manually set the batch size.
        if batch_size is not None:
            dl_cfg["batch_size"] =  batch_size
        
        self.train_dl = DataLoader(
            self.train_dataset, 
            shuffle=True, 
            **dl_cfg
        )
        self.val_dl = DataLoader(
            self.val_dataset, 
            shuffle=False, 
            drop_last=False, 
            **dl_cfg
        )

    def build_model(self):

        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_cfg_dict = self.config.to_dict()

        #######################
        # LOAD THE EXPERIMENT #
        #######################
        # Get the configs of the experiment
        load_exp_args = {
            "checkpoint": total_cfg_dict['train'].get('base_checkpoint', 'max-val-dice_score'),
            "exp_kwargs": {
                "set_seed": True, # Important, we want to use the same seed.
                "load_data": False, # Important, we might want to modify the data construction.
                "load_aug_pipeline": False, # Important, we might want to modify the augmentation pipeline.
            }
        }
            
        # Backwards compatibility for the pretrained directory.
        base_pt_key = 'pretrained_dir' if 'base_pretrained_dir' not in total_cfg_dict['train']\
            else 'base_pretrained_dir'
        # Either select from a set of experiments in a common directory OR choose a particular experiment to load.
        self.pt_model_path = total_cfg_dict['train'][base_pt_key]
        if "config.yml" in os.listdir(self.pt_model_path):
            self.pretrained_exp = load_experiment(
                path=self.pt_model_path,
                **load_exp_args
            )
        else:
            rs = ResultsLoader()
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(rs.load_configs(self.pt_model_path, properties=False)),
                selection_metric=total_cfg_dict['train']['base_pt_select_metric'],
                **load_exp_args
            )
        # Send the pretrained experiment to the device.
        self.pretrained_exp.to_device()

        # Now we can access the old total config. 
        pt_exp_cfg_dict = self.pretrained_exp.config.to_dict()

        #######################################
        #  Add any preprocessing augs from pt #
        #######################################
        if ('augmentations' in pt_exp_cfg_dict.keys()) and\
            total_cfg_dict['train'].get('use_pretrained_norm_augs', False):
            flat_exp_aug_cfg = valmap(list2tuple, HDict(pt_exp_cfg_dict['augmentations']).flatten())
            norm_augs = {exp_key: exp_val for exp_key, exp_val in flat_exp_aug_cfg.items() if 'normalize' in exp_key}
            # If the pretrained experiment used normalization augmentations, then add them to the new experiment.``
            if norm_augs != {}:
                if ('augmentations' in total_cfg_dict.keys()):
                    if 'visual' in total_cfg_dict['augmentations'].keys():
                        total_cfg_dict['augmentations']['visual'].update(norm_augs)
                    else:
                        total_cfg_dict['augmentations']['visual'] = norm_augs
                else:
                    total_cfg_dict['augmentations'] = {
                        'visual': norm_augs,
                    }
        
        #########################################
        #            Model Creation             #
        #########################################
        # Either keep training the network, or use a post-hoc calibrator.
        model_cfg_dict = total_cfg_dict['model']
        self.model_class = model_cfg_dict['_class']
        if self.model_class is None:
            self.base_model = torch.nn.Identity() # Therh is no learned calibrator.
            self.model = self.pretrained_exp.model
            # Edit the model_config.
            total_cfg_dict['model']['_class'] = parse_class_name(str(self.base_model.__class__))
        else:
            # Get the pretrained model out of the old experiment.
            self.base_model = self.pretrained_exp.model

            # Prepare the pretrained model.
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
    
            self.model = eval_config(Config(model_cfg_dict))
            # If the model has a weights_init method, call it to initialize the weights.
            if hasattr(self.model, "weights_init"):
                self.model.weights_init()
            # Edit the model_config, note that this is flipped with above.
            total_cfg_dict['model']['_class'] = parse_class_name(str(self.model.__class__))

        ########################################################################
        # Make sure we use the old experiment seed and add important metadata. #
        ########################################################################
        # Get the tuned calibration parameters.
        self.properties["num_params"] = num_params(self.model)

        # Set the new experiment params as the old ones.
        old_exp_cfg = pt_exp_cfg_dict['experiment']
        old_exp_cfg.update(total_cfg_dict['experiment'])
        new_exp_cfg = old_exp_cfg.copy()
        total_cfg_dict['experiment'] = new_exp_cfg

        # Save the new config because we edited it and reset self.config
        autosave(total_cfg_dict, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_cfg_dict)
        self.to_device()

        # Compile optimizes our run speed by fusing operations.
        if self.config['experiment'].get('torch_compile', False):
            self.base_model = torch.compile(self.base_model)
            if self.model_class is not None:
                self.model = torch.compile(self.model)

        # If there is a pretrained model, load it.
        train_config = total_cfg_dict['train']
        if ("pretrained_dir" in train_config) and\
            (total_cfg_dict.get('experiment', {}).get("restart", False)):
            checkpoint_dir = f'{train_config["pretrained_dir"]}/checkpoints/{train_config["load_chkpt"]}.pt'
            # Load the checkpoint dir and set the model to the state dict.
            checkpoint = torch.load(checkpoint_dir, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])

    def build_optim(self):
        optim_cfg_dict = self.config["optim"].to_dict()
        train_cfg_dict = self.config["train"].to_dict()
        exp_cfg_dict = self.config.get("experiment", {}).to_dict()

        if 'lr_scheduler' in optim_cfg_dict:
            self.lr_scheduler = eval_config(optim_cfg_dict.pop('lr_scheduler', None))

        if "weight_decay" in optim_cfg_dict:
            optim_cfg_dict["params"] = split_param_groups_by_weight_decay(
                self.model, optim_cfg_dict["weight_decay"]
            )
        else:
            optim_cfg_dict["params"] = self.model.parameters()

        self.optim = eval_config(optim_cfg_dict)

        # If there is a pretrained model, then load the optimizer state.
        if "pretrained_dir" in train_cfg_dict and exp_cfg_dict.get("restart", False):
            checkpoint_dir = f'{train_cfg_dict["pretrained_dir"]}/checkpoints/{train_cfg_dict["load_chkpt"]}.pt'
            # Load the checkpoint dir and set the model to the state dict.
            checkpoint = torch.load(checkpoint_dir, map_location=self.device, weights_only=True)
            self.optim.load_state_dict(checkpoint["optim"])
        else:
            # Zero out the gradients as initialization 
            self.optim.zero_grad()
        
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
            # TODO: BACKWARDS COMPATIBILITY STOPGAP
            loss_cfg = self.config["loss_func"].to_dict()
            loss_cfg["_class"] = loss_cfg["_class"].replace("ese.experiment", "ese")
            self.loss_func = eval_config(Config(loss_cfg))
    
    def run_step(self, batch_idx, batch, backward, augmentation, **kwargs):
        # Send data and labels to device.
        batch = to_device(batch, self.device)

        # Get the image and label.
        x, y = batch["img"], batch["label"]

        if 'data_id' in batch:
            print(f"Data ID: {batch['data_id']}")

        # Apply the augmentation on the GPU.
        if augmentation:
            with torch.no_grad():
                # For now only the image gets augmented.
                x, y = self.aug_pipeline(x, y)
        
        # Zero out the gradients.
        self.optim.zero_grad()

        if self.config['experiment'].get('torch_mixed_precision', False):
            # Run the forward and loss computation with autocast.
            with autocast('cuda'):
                y_hat, loss = self.run_forward(x, y)
            # If backward then backprop the gradients.
            if backward:
                # Scale the loss and backpropagate
                self.grad_scaler.scale(loss).backward()
                # Step the optimizer using the scaler
                self.grad_scaler.step(self.optim)
                # Update the scale for next iteration
                self.grad_scaler.update() 
        else:
            # Run the forward pass.
            y_hat, loss = self.run_forward(x, y)
            # If backward then backprop the gradients.
            if backward:
                loss.backward()
                self.optim.step()
        
        # For the ground truth segmentation, if 'gt_seg' is in our batch,
        # we will use that, otherwise we will use the 'label'.
        forward_batch = {
            "x": x,
            "y_true": batch.get('gt_seg', y),
            "loss": loss,
            "y_pred": y_hat, # Used for visualization functions.
            "batch_idx": batch_idx,
            "from_logits": True
        }
        self.run_callbacks("step", batch=forward_batch)

        return forward_batch
    
    def run_forward(self, x, y):
        with torch.no_grad():
            y_hat_uncal = self.base_model(x)

        # Calibrate the predictions.
        y_hat, y_hat_temps = self.model(y_hat_uncal, image=x)

        # Get the target type and then calculate the loss for the necessary quantity.
        target = self.config["data"].get("target", "seg")
        if target == "seg":
            loss = self.loss_func(y_hat, y)
        elif target == "temp":
            print("Predicted Batch Temps:", y_hat_temps, "with shape:", y_hat_temps.shape)
            print("GT Temps:", y, "with shape:", y.shape)
            loss = self.loss_func(y_hat_temps, y)
            print("Loss:", loss)
            print()
        else:
            raise ValueError(f"Target type {target} not recognized.")

        return y_hat, loss

    def to_device(self):
        self.base_model = to_device(self.base_model, self.device, channels_last=False)
        self.model = to_device(self.model, self.device, channels_last=False)

    def run(self):
        print(f"Running {str(self)}")
        epochs: int = self.config["train.epochs"]

        # If using mixed precision, then create a GradScaler to scale gradients during mixed precision training.
        if self.config.get('experiment.torch_mixed_precision', False):
            self.grad_scaler = GradScaler('cuda')

        self.build_dataloader()
        self.build_callbacks()

        last_epoch: int = self.properties.get("epoch", -1)
        if last_epoch >= 0:
            self.load(tag="last")
            df = self.metrics.df
            autosave(df[df.epoch < last_epoch], self.path / "metrics.jsonl")
        else:
            self.build_initialization()

        checkpoint_freq: int = self.config.get("log.checkpoint_freq", 1)
        eval_freq: int = self.config.get("train.eval_freq", 1)

        for epoch in range(last_epoch + 1, epochs):
            self._epoch = epoch

            # Either we run a validation epoch first and then do a round of training...
            if not self.config['experiment'].get('val_first', False):
                print(f"Start training epoch {epoch}.")
                self.run_phase("train", epoch)

            # Evaluate the model on the validation set.
            if eval_freq > 0 and (epoch % eval_freq == 0 or epoch == epochs - 1):
                print(f"Start validation round at {epoch}.")
                self.run_phase("val", epoch)

            # ... or we run a training epoch first and then do a round of validation.
            if self.config['experiment'].get('val_first', False):
                print(f"Start training epoch {epoch}.")
                self.run_phase("train", epoch)

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.checkpoint()

            self.run_callbacks("epoch", epoch=epoch)

        self.checkpoint(tag="last")
        self.run_callbacks("wrapup")

    def predict(
        self, 
        x, 
        multi_class: bool,
        threshold: float = 0.5,
        temperature: Optional[float] = None,
        label: Optional[int] = None,
    ):
        # Predict with the base model.
        with torch.no_grad():
            uncal_yhat = self.base_model(x)

        # Apply post-hoc calibration.
        if self.model_class is None:
            cal_logit_map = self.model(uncal_yhat)
        else:
            cal_logit_map = self.model(uncal_yhat, image=x)

        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            cal_logit_map, 
            from_logits=True,
            threshold=threshold,
            temperature=temperature,
            multi_class=multi_class
        )

        # If label is not None, then only return the predictions for that label.
        if label is not None:
            cal_logit_map = cal_logit_map[:, label, ...].unsqueeze(1)
            prob_map = prob_map[:, label, ...].unsqueeze(1)
        
        # Assert that the hard prediction is unchanged.
        assert (pred_map == (torch.sigmoid(uncal_yhat)> 0.5)).all(),\
            "The hard prediction should not change after calibration."

        # Return the outputs
        return {
            'y_logits': cal_logit_map,
            'y_probs': prob_map, 
            'y_hard': pred_map 
        }
