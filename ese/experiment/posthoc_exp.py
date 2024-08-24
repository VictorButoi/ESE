# local imports
from ..augmentation.gather import augmentations_from_config
from .utils import load_experiment, process_pred_map, parse_class_name
# torch imports
import torch
from torch.utils.data import DataLoader
# IonPy imports
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.hash import json_digest
from ionpy.analysis import ResultsLoader
from ionpy.util.torchutils import to_device
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.nn.util import num_params, split_param_groups_by_weight_decay
# misc imports
import os
from typing import Optional
import matplotlib.pyplot as plt


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
        # TODO: BACKWARDS COMPATIBILITY STOPGAP
        dataset_cls_name = new_data_cfg.pop("_class").replace("ese.experiment", "ese")

        dataset_cls = absolute_import(dataset_cls_name)
        # Build the augmentation pipeline.
        augmentation_list = total_config.get("augmentations", None)
        if augmentation_list is not None:
            train_transforms = augmentations_from_config(augmentation_list.get("train", None))
            val_transforms = augmentations_from_config(augmentation_list.get("val", None))
            self.properties["aug_digest"] = json_digest(augmentation_list)[:8]
        else:
            train_transforms, val_transforms = None, None

        if load_data:
            train_split = new_data_cfg.pop("train_splits", None)
            val_split = new_data_cfg.pop("val_splits", None)
            num_examples = new_data_cfg.pop("num_examples", None)

            splits_defined = train_split is not None and val_split is not None

            # Initialize the dataset classes.
            self.train_dataset = dataset_cls(
                split=train_split if splits_defined else "train",
                transforms=train_transforms, 
                num_examples=num_examples,
                **new_data_cfg
            )
            self.val_dataset = dataset_cls(
                split=val_split if splits_defined else "val",
                transforms=val_transforms, 
                **new_data_cfg
            )

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
            "checkpoint": total_cfg_dict['train'].get('base_checkpoint', 'max-val-dice_score')
        }
            
        # Either select from a set of experiments in a common directory OR choose a particular experiment to load.
        if "config.yml" in os.listdir(total_cfg_dict['train']['base_pretrained_dir']):
            self.pretrained_exp = load_experiment(
                path=total_cfg_dict['train']['base_pretrained_dir'],
                **load_exp_args
            )
        else:
            rs = ResultsLoader()
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(rs.load_configs(total_cfg_dict['train']['base_pretrained_dir'], properties=False)),
                selection_metric=total_cfg_dict['train']['base_pt_select_metric'],
                **load_exp_args
            )
        # Now we can access the old total config. 
        pretrained_total_cfg_dict = self.pretrained_exp.config.to_dict()

        #########################################
        #            Model Creation             #
        #########################################
        train_config = total_cfg_dict['train']
        model_cfg_dict = total_cfg_dict['model']
        pretrained_model_cfg_dict = pretrained_total_cfg_dict['model']
        # Either keep training the network, or use a post-hoc calibrator.
        self.model_class = model_cfg_dict['_class']
        if self.model_class is None:
            self.base_model = torch.nn.Identity() # Therh is no learned calibrator.
            self.model = self.pretrained_exp.model
            # Edit the model_config.
            total_cfg_dict['model']['_class'] = parse_class_name(str(self.base_model.__class__))
        else:
            self.base_model = self.pretrained_exp.model
            self.base_model.eval()
            # Get the old in and out channels from the pretrained model.
            # TODO: Kind of hardcoded to binary greyscale segmentation...
            model_cfg_dict["num_classes"] = pretrained_model_cfg_dict.get('out_channels', 1)
            model_cfg_dict["image_channels"] = pretrained_model_cfg_dict.get('in_channels', 1)

            # TODO: BACKWARDS COMPATIBILITY STOPGAP
            model_cfg_dict["_class"] = model_cfg_dict["_class"].replace("ese.experiment", "ese")

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
        total_cfg_dict['experiment'] = pretrained_total_cfg_dict['experiment']
        # Save the new config because we edited it and reset self.config
        autosave(total_cfg_dict, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_cfg_dict)

        # If there is a pretrained model, load it.
        if "pretrained_dir" in train_config:
            checkpoint_dir = f'{train_config["pretrained_dir"]}/checkpoints/{train_config["load_chkpt"]}.pt'
            # Load the checkpoint dir and set the model to the state dict.
            checkpoint = torch.load(checkpoint_dir, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
        
        # Put the model on the device here.
        self.to_device()

    def build_optim(self):
        optim_cfg_dict = self.config["optim"].to_dict()
        train_cfg_dict = self.config["train"].to_dict()

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
        if "pretrained_dir" in train_cfg_dict:
            checkpoint_dir = f'{train_cfg_dict["pretrained_dir"]}/checkpoints/{train_cfg_dict["load_chkpt"]}.pt'
            # Load the checkpoint dir and set the model to the state dict.
            checkpoint = torch.load(checkpoint_dir, map_location=self.device)
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
    
    def run_step(self, batch_idx, batch, backward, **kwargs):
        # Send data and labels to device.
        batch = to_device(batch, self.device)

        # Get the image and label from the batch.
        x = batch["img"]
        y = batch["label"]

        # Forward pass
        with torch.no_grad():
            yhat = self.base_model(x)
        
        # plt.imshow(yhat[0, 0].cpu().numpy(), cmap='gray', interpolation='none')
        # plt.colorbar()
        # plt.show()

        # Calibrate the predictions.
        if self.model_class is None:
            yhat_cal = self.model(yhat)
        else:
            yhat_cal = self.model(yhat, image=x)

        # plt.imshow(yhat_cal[0, 0].cpu().detach().numpy(), cmap='gray', interpolation='none')
        # plt.colorbar()
        # plt.show()

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
