# local imports
from .utils import load_experiment, process_pred_map, parse_class_name, filter_args_by_class
from ..augmentation.pipeline import build_aug_pipeline
from ..augmentation.gather import augmentations_from_config
# torch imports
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch._dynamo # For compile
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


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def calculate_tensor_memory_in_gb(tensor):
    # Get the number of elements in the tensor
    num_elements = tensor.numel()
    # Get the size of each element in bytes based on the dtype
    dtype_size = tensor.element_size()  # size in bytes for the tensor's dtype
    # Total memory in bytes
    total_memory_bytes = num_elements * dtype_size
    # Convert bytes to gigabytes (1 GB = 1e9 bytes)
    total_memory_gb = total_memory_bytes / 1e9
    
    return total_memory_gb


class PostHocExperiment(TrainExperiment):

    def build_augmentations(self, load_aug_pipeline):
        super().build_augmentations()
        if "augmentations" in self.config and load_aug_pipeline:
            self.aug_pipeline = build_aug_pipeline(self.config.to_dict()["augmentations"])

    def build_data(self, load_data):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        new_data_cfg_dict = total_config.get("data", {})
        # Get the data and transforms we want to apply
        pretrained_data_cfg = self.pretrained_exp.config["data"].to_dict()
        # Update the old cfg with new cfg (if it exists).
        pretrained_data_cfg.update(new_data_cfg_dict)
        new_data_cfg = pretrained_data_cfg.copy()
        # Finally update the data config with the copy. 
        total_config["data"] = new_data_cfg 

         # Save the new config because we edited it.
        autosave(total_config, self.path / "config.yml")
        self.config = Config(total_config)

        # Get the dataset class and build the transforms
        # TODO: BACKWARDS COMPATIBILITY STOPGAP
        dataset_cls_name = new_data_cfg.pop("_class").replace("ese.experiment", "ese")
        dataset_cls = absolute_import(dataset_cls_name)

        # Build the augmentation pipeline.
        if new_data_cfg_dict != {} and new_data_cfg.get("augmentations", None):
            augmentation_list = new_data_cfg.get("augmentations", None)
            train_transforms = augmentations_from_config(augmentation_list.get("train", None))
            val_transforms = augmentations_from_config(augmentation_list.get("val", None))
            self.properties["aug_digest"] = json_digest(augmentation_list)[:8]
        else:
            train_transforms, val_transforms = None, None

        if load_data:
            # If we are limiting the number of examples, then pop the number of examples.   
            num_examples = new_data_cfg.pop("num_examples", None)

            # If we are using specific examples, then pop the examples.
            train_examples = new_data_cfg.pop("train_examples", None)
            val_examples = new_data_cfg.pop("val_examples", None)

            # We need to filter the arguments that are not needed for the dataset class.
            filtered_new_data_cfg = filter_args_by_class(dataset_cls, new_data_cfg)

            # Initialize the dataset classes.
            self.train_dataset = dataset_cls(
                split=new_data_cfg["train_splits"],
                transforms=train_transforms, 
                examples=train_examples,
                num_examples=num_examples,
                **filtered_new_data_cfg
            )
            self.val_dataset = dataset_cls(
                split=new_data_cfg["val_splits"],
                transforms=val_transforms, 
                examples=val_examples,
                **filtered_new_data_cfg
            )

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
            "device": "cuda",
            "checkpoint": total_cfg_dict['train'].get('base_checkpoint', 'max-val-dice_score'),
            "exp_kwargs": {
                "load_data": False, # Important, we might want to modify the data construction.
                "load_aug_pipeline": False, # Important, we might want to modify the augmentation pipeline.
                "set_seed": True, # Important, we want to use the same seed.
            }
        }
            
        # Backwards compatibility for the pretrained directory.
        base_pt_key = 'pretrained_dir' if 'base_pretrained_dir' not in total_cfg_dict['train']\
            else 'base_pretrained_dir'
        # Either select from a set of experiments in a common directory OR choose a particular experiment to load.
        if "config.yml" in os.listdir(total_cfg_dict['train'][base_pt_key]):
            self.pretrained_exp = load_experiment(
                path=total_cfg_dict['train'][base_pt_key],
                **load_exp_args
            )
        else:
            rs = ResultsLoader()
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(rs.load_configs(total_cfg_dict['train'][base_pt_key], properties=False)),
                selection_metric=total_cfg_dict['train']['base_pt_select_metric'],
                **load_exp_args
            )
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

            # Get the old in and out channels from the pretrained model.
            # TODO: Kind of hardcoded to binary greyscale segmentation...
            model_cfg_dict["num_classes"] = pt_exp_cfg_dict['model'].get('out_channels', 1)
            model_cfg_dict["image_channels"] = pt_exp_cfg_dict['model'].get('in_channels', 1)
    
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
        start_time = time.time()

        # Send data and labels to device.
        batch = to_device(batch, self.device)

        # Get the image and label.
        if isinstance(batch, dict):
            x, y = batch["img"], batch["label"]
        else:
            x, y = batch[0], batch[1]

        # Apply the augmentation on the GPU.
        if augmentation:
            with torch.no_grad():
                # For now only the image gets augmented.
                x, y = self.aug_pipeline(x, y)
        
        # Zero out the gradients.
        self.optim.zero_grad()

        if self.config['experiment'].get('torch_mixed_precision', False):
            with autocast('cuda'):
                with torch.no_grad():
                    yhat_uncal = self.base_model(x)        
                # Calibrate the predictions.
                if self.model_class is None:
                    yhat_cal = self.model(yhat_uncal)
                else:
                    yhat_cal = self.model(yhat_uncal, image=x)
                # Calculate the loss between the pred and original preds.
                loss = self.loss_func(yhat_cal, y)

            # If backward then backprop the gradients.
            if backward:
                # Scale the loss and backpropagate
                self.grad_scaler.scale(loss).backward()
                # Step the optimizer using the scaler
                self.grad_scaler.step(self.optim)
                # Update the scale for next iteration
                self.grad_scaler.update() 
        else:
            with torch.no_grad():
                yhat_uncal = self.base_model(x)

            # Calibrate the predictions.
            if self.model_class is None:
                yhat_cal = self.model(yhat_uncal)
            else:
                yhat_cal = self.model(yhat_uncal, image=x)

            # Calculate the loss between the pred and original preds.
            loss = self.loss_func(yhat_cal, y)
            # If backward then backprop the gradients.
            if backward:
                loss.backward()
                self.optim.step()

        # Run step-wise callbacks if you have them.
        forward_batch = {
            "x": x,
            "y_true": y,
            "y_pred": yhat_cal,
            "y_logits": yhat_cal, # Used for visualization functions.
            "loss": loss,
            "batch_idx": batch_idx
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
            uncal_yhat = self.base_model(x)

        # Apply post-hoc calibration.
        if self.model_class is None:
            cal_logit_map = self.model(uncal_yhat)
        else:
            cal_logit_map = self.model(uncal_yhat, image=x)

        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            cal_logit_map, 
            multi_class=multi_class, 
            threshold=threshold,
            from_logits=True
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

