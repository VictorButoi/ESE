# local imports
from .utils import process_pred_map
from ..models.ensemble import get_combine_fn
# torch imports
import torch
# IonPy imports
from ionpy.util import Config
from ionpy.analysis import ResultsLoader
from ionpy.experiment import BaseExperiment
from ionpy.datasets.cuda import CUDACachedDataset
from ionpy.experiment.util import absolute_import
from ionpy.util.ioutil import autosave
# misc imports
import json
from pathlib import Path


# Very similar to BaseExperiment, but with a few changes.
class EnsembleInferenceExperiment(BaseExperiment):

    def __init__(self, path, set_seed=True, load_data=False):
        torch.backends.cudnn.benchmark = True
        super().__init__(path, set_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.build_data(load_data)
    
    def build_data(self, load_data):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        # Update the old cfg with new cfg (if it exists).
        if "data" in self.config:
            self.pretrained_data_cfg.update(self.config["data"].to_dict())
        total_config["data"] = self.pretrained_data_cfg
        autosave(total_config, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_config)

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
                if column not in ["seed", "path", "pretrained_dir"] and not dfc[column].nunique() == 1:
                    raise ValueError(f"The only difference between the configs should be the seed, but found different values in column '{column}'.")
        # Verify that the configs are valid.
        verify_ensemble_configs(dfc)
        # Loop through each config and build the experiment, placing it in a dictionary.
        self.ens_exp_paths = []
        self.ens_exps = {}
        total_params = 0
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
            self.ens_exp_paths.append(exp_path)
            self.ens_exps[exp_path] = loaded_exp
            total_params += loaded_exp.properties["num_params"]
            # Set the pretrained data config from the first model.
            if exp_idx == 0:
                self.pretrained_data_cfg = loaded_exp.config["data"].to_dict()

        self.num_params = total_params
        # Build the combine function.
        self.combine_fn = get_combine_fn(model_cfg["ensemble_combine_fn"])
        self.properties["num_params"] = self.num_params 

    def to_device(self):
        for exp_path in self.ens_exp_paths:
            self.ens_exps[exp_path].to_device()

    def predict(self, x, multi_class, threshold=0.5):
        # Get the label predictions for each model.
        ensemble_model_outputs = {}
        for exp_path in self.ens_exp_paths:
            # Multi-class needs to be true here so that we can combine the outputs.
            ensemble_model_outputs[exp_path] = self.ens_exps[exp_path].predict(
                x=x, multi_class=True, return_logits=True
            )['ypred']
        #Get the model cfg
        model_cfg = self.config["model"].to_dict()
        # Combine the outputs of the models.
        prob_map = self.combine_fn(
            ensemble_model_outputs, 
            pre_softmax=model_cfg["ensemble_pre_softmax"]
            )
        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            prob_map, 
            multi_class=multi_class, 
            threshold=threshold,
            from_logits=False # The combine_fn already returns probs.
            )
        # Return the outputs
        return {
            'ypred': prob_map, 
            'yhard': pred_map 
        }