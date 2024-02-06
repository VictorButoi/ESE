# local imports
from .utils import process_pred_map, parse_class_name
from ..models.ensemble_utils import get_combine_fn, get_ensemble_member_weights
# torch imports
import torch
from torch import Tensor
# IonPy imports
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.analysis import ResultsLoader
from ionpy.experiment import BaseExperiment
from ionpy.datasets.cuda import CUDACachedDataset
from ionpy.experiment.util import absolute_import
# misc imports
import json
from pathlib import Path
from typing import Optional, Literal


# Very similar to BaseExperiment, but with a few changes.
class BinningInferenceExperiment(BaseExperiment):

    def __init__(self, path, set_seed=True, load_data=False):
        torch.backends.cudnn.benchmark = True
        super().__init__(path, set_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.build_data(load_data)
        # Save the config because we've modified it.
        autosave(self.config.to_dict(), self.path / "config.yml") # Save the new config because we edited it.
    
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
        # Optionally, load the data.
        if load_data:
            # Get the dataset class and build the transforms
            dataset_cls = absolute_import(pretrained_data_cfg.pop("_class"))

            # Build the augmentation pipeline.
            if "augmentations" in total_config and (total_config["augmentations"] is not None):
                val_transforms = augmentations_from_config(total_config["augmentations"]["train"])
                cal_transforms = augmentations_from_config(total_config["augmentations"]["val"])
                self.properties["aug_digest"] = json_digest(self.config["augmentations"].to_dict())[
                    :8
                ]
            else:
                val_transforms, cal_transforms = None, None
            # Build the datasets, apply the transforms
            self.train_dataset = dataset_cls(split="val", transforms=val_transforms, **pretrained_data_cfg)
            self.val_dataset = dataset_cls(split="cal", transforms=cal_transforms, **pretrained_data_cfg)

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

    def build_model(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        ###################
        # BUILD THE MODEL #
        ###################
        # Get the configs of the experiment
        load_exp_cfg = {
            "device": "cuda",
            "load_data": False, # Important, we might want to modify the data construction.
        }
        if "config.yml" in os.listdir(total_config['train']['pretrained_dir']):
            self.pretrained_exp = load_experiment(
                path=total_config['train']['pretrained_dir'],
                **load_exp_cfg
            )
        else:
            rs = ResultsLoader()
            dfc = rs.load_configs(
                total_config['train']['pretrained_dir'],
                properties=False,
            )
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(dfc),
                selection_metric=total_config['train']['pretrained_select_metric'],
                **load_exp_cfg
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
        if self.model_class == "Vanilla":
            self.base_model = torch.nn.Identity()
            # Load the model, there is no learned calibrator.
            self.model = self.pretrained_exp.model
            self.properties["num_params"] = num_params(self.model)
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
            self.properties["num_params"] = num_params(self.model) + num_params(self.base_model)
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
        self.loss_func = eval_config(self.config["loss_func"])

    def predict(self, 
                x, 
                multi_class,
                threshold=0.5,
                return_logits=False):
        assert x.shape[0] == 1, "Batch size must be 1 for prediction for now."
        # Predict with the base model.
        with torch.no_grad():
            yhat = self.base_model(x)
        # Apply post-hoc calibration.
        if self.model_class == "Vanilla":
            yhat_cal = self.model(yhat)
        else:
            yhat_cal = self.model(yhat, image=x)
        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            yhat_cal, 
            multi_class=multi_class, 
            threshold=threshold,
            return_logits=return_logits
            )
        # Return the outputs
        return {
            'y_pred': prob_map, 
            'y_hard': pred_map 
        }

    def to_device(self):
        self.base_model = to_device(self.base_model, self.device, channels_last=False)
        self.model = to_device(self.model, self.device, channels_last=False)
