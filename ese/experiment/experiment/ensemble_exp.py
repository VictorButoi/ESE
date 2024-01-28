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
class EnsembleInferenceExperiment(BaseExperiment):

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
        # Update the old cfg with new cfg (if it exists).
        if "data" in self.config:
            self.pretrained_data_cfg.update(self.config["data"].to_dict())
        # Set the data config as the pretrained data config.
        total_config["data"] = self.pretrained_data_cfg
        self.config = Config(total_config)
        # Optionally, load the data.
        if load_data:
            data_cfg = self.config["data"].to_dict()
            # Get the dataset class and build the transforms
            dataset_cls = absolute_import(data_cfg.pop("_class"))
            if "cuda" in data_cfg:
                assert data_cfg["preload"], "If you want to cache the dataset on the GPU, you must preload it."
                cache_dsets_on_gpu = data_cfg.pop("cuda")
            else:
                cache_dsets_on_gpu = False
            # Build the datasets, apply the transforms
            self.train_dataset = dataset_cls(split="val", **data_cfg)
            self.val_dataset = dataset_cls(split="cal", **data_cfg)
            # Optionally cache the datasets on the GPU.
            if cache_dsets_on_gpu:
                self.train_dataset = CUDACachedDataset(self.train_dataset)
                self.val_dataset = CUDACachedDataset(self.val_dataset)

    def build_model(self):
        # Use the total config
        total_config = self.config.to_dict()
        # Use the model config.
        model_cfg = total_config["model"]
        # Get the configs of the experiment
        rs = ResultsLoader()
        # Load ALL the configs in the directory.
        dfc = rs.load_configs(
            model_cfg["pretrained_exp_root"],
            properties=False,
        )
        def verify_ensemble_configs(dfc):
            seed_values = dfc["seed"].unique()
            unique_runs = dfc["path"].unique()
            if len(seed_values) < len(unique_runs):
                raise ValueError("Duplicate seeds found in ensemble.")
            for column in dfc.columns:
                if column not in ["seed", "path", "pretrained_dir"] and\
                    len(dfc[column].unique()) > 1:
                    raise ValueError(f"The only difference between the configs should be the seed, "\
                                      + f"but found different values in column '{column}'."\
                                        + f"Unique values: {dfc[column].unique()}")
        # Verify that the configs are valid.
        verify_ensemble_configs(dfc)
        # Build the combine function.
        if "ensemble_cfg" in model_cfg:
            self.ensemble_combine_fn = model_cfg["ensemble_cfg"][0]
            self.ensemble_combine_quantity = model_cfg["ensemble_cfg"][1]
        else:
            self.ensemble_combine_fn = None
            self.ensemble_combine_quantity = None
        # Loop through each config and build the experiment, placing it in a dictionary.
        self.ens_exp_paths = []
        self.ens_exps = {}
        self.num_params = 0
        # Loop through each member of the ensemble.
        for exp_idx, exp_path in enumerate(dfc["path"].unique()):
            # Get the experiment class
            properties_dir = Path(exp_path) / "properties.json"
            with open(properties_dir, 'r') as prop_file:
                props = json.load(prop_file)
            exp_class = absolute_import(f'ese.experiment.experiment.{props["experiment"]["class"]}')

            # Load the experiment
            loaded_exp = exp_class(exp_path, load_data=False)
            if model_cfg["checkpoint"] is not None:
                loaded_exp.load(tag=model_cfg["checkpoint"])
            loaded_exp.model.eval()

            self.ens_exps[str(exp_path)] = loaded_exp
            self.num_params += loaded_exp.properties["num_params"]
            self.ens_exp_paths.append(str(exp_path))

            # Set the pretrained data config from the first model.
            if exp_idx == 0:
                self.pretrained_data_cfg = loaded_exp.config["data"].to_dict()
                # Figure out what the model and potentially the pretrained backbone are.
                if hasattr(loaded_exp, "base_model"):
                    # Do some bookkeping about the kinds of models we are including in the ensemble.
                    main_model_name = parse_class_name(str(loaded_exp.model.__class__))
                    pretrained_model_name = parse_class_name(str(loaded_exp.base_model.__class__))
                    # If the pretrained model_name is identity, then we need to flip the names.
                    if pretrained_model_name.split('.')[-1] == "Identity":
                        temp = main_model_name
                        main_model_name = pretrained_model_name
                        pretrained_model_name = temp 
                else:
                    main_model_name = parse_class_name(str(loaded_exp.model.__class__))
                    pretrained_model_name = None
                # Add the model class to the config.
                model_cfg["_class"] = main_model_name
                model_cfg["_pretrained_class"] = pretrained_model_name
        ################################################
        # Get the weights per ensemble member.
        ################################################
        self.ens_mem_weights = get_ensemble_member_weights(
            results_df=rs.load_metrics(dfc),
            metric=model_cfg["ensemble_w_metric"]
        )
        ####################################################
        # Add other auxilliary information to the config.
        ####################################################
        # Count the number of parameters.
        self.properties["num_params"] = self.num_params 
        # Create the new config.
        total_config["model"] = model_cfg
        self.config = Config(total_config)

    def to_device(self):
        for exp_path in self.ens_exp_paths:
            self.ens_exps[exp_path].to_device()
        # Move the weights to the device.
        self.ens_mem_weights = self.ens_mem_weights.to(self.device)

    def predict(
        self, 
        x: torch.Tensor, 
        multi_class: bool, 
        threshold: float = 0.5, 
        weights: Optional[Tensor] = None,
        combine_fn: Optional[str] = None,
        combine_quantity: Optional[Literal["probs", "logits"]] = None
    ):
        # Get the label predictions for each model.
        ensemble_model_outputs = {}
        for exp_path in self.ens_exp_paths:
            # Multi-class needs to be true here so that we can combine the outputs.
            ensemble_model_outputs[exp_path] = self.ens_exps[exp_path].predict(
                x=x, multi_class=True, return_logits=True
            )['y_pred']
        # Combine the outputs of the models.
        if combine_fn is None:
            assert self.ensemble_combine_fn is not None, "No combine function provided."
            combine_fn = self.ensemble_combine_fn
        if combine_quantity is None:
            assert self.ensemble_combine_quantity is not None, "No pre_softmax value provided."
            combine_quantity = self.ensemble_combine_quantity
        if weights is None:
            assert self.ens_mem_weights is not None, "No weights provided."   
            weights = self.ens_mem_weights
        # Combine the outputs of the models.
        prob_map = get_combine_fn(combine_fn)(
            ensemble_model_outputs, 
            combine_quantity=combine_quantity,
            weights=weights
        )
        # Get the hard prediction and probabilities, if we are doing identity,
        # then we don't want to return probs.
        prob_map, pred_map = process_pred_map(
            prob_map, 
            multi_class=multi_class, 
            threshold=threshold,
            from_logits=False, # Ensemble methods already return probs.
            return_logits=(combine_fn == "identity")
            )
        # Return the outputs
        return {
            'y_pred': prob_map, 
            'y_hard': pred_map # if identity, this will be None.
        }