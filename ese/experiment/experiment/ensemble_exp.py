# local imports
from .utils import process_logits_map
from ..models.ensemble import get_combine_fn
# torch imports
import torch
# IonPy imports
from ionpy.nn.util import num_params
from ionpy.analysis import ResultsLoader
from ionpy.util.torchutils import to_device
from ionpy.experiment import BaseExperiment
from ionpy.experiment.util import absolute_import
# misc imports
import json
from pathlib import Path
from typing import Optional


# Very similar to BaseExperiment, but with a few changes.
class EnsembleExperiment(BaseExperiment):

    def __init__(self, path, set_seed=True, build_data=False):
        torch.backends.cudnn.benchmark = True
        super().__init__(path, set_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        # Use the model config.
        model_cfg = self.config["model"].to_dict()
        # Get the configs of the experiment
        rs = ResultsLoader()
        # Load ALL the configs in the directory.
        dfc = rs.load_configs(
            model_cfg["pretrained_exp_root"],
            properties=False,
        )
        def check_configs(dfc):
            seed_values = dfc["seed"].unique()
            unique_runs = dfc["path"].unique()
            if len(seed_values) < len(unique_runs):
                raise ValueError("Duplicate seeds found in ensemble.")
            for column in dfc.columns:
                if column not in ["seed", "path"] and not dfc[column].nunique() == 1:
                    raise ValueError(f"The only difference between the configs should be the seed, but found different values in column '{column}'.")
        # Verify that the configs are valid.
        check_configs(dfc)
        # Loop through each config and build the experiment, placing it in a dictionary.
        self.paths = []
        self.models= {}
        total_params = 0
        for exp_path in dfc["path"].unique():
            # Get the experiment class
            print(exp_path)
            properties_dir = Path(exp_path) / "properties.json"
            with open(properties_dir, 'r') as prop_file:
                props = json.load(prop_file)
            exp_class = absolute_import(f'ese.experiment.experiment.{props["experiment"]["class"]}')
            # Load the experiment
            print(exp_class)
            loaded_exp = exp_class(exp_path, build_data=False)
            if model_cfg["checkpoint"] is not None:
                loaded_exp.load(tag=model_cfg["checkpoint"])
            loaded_exp.model.eval()
            self.paths.append(exp_path)
            self.models[exp_path] = loaded_exp.model
            total_params += num_params(loaded_exp.model)
        self.num_params = total_params
        # Build the combine function.
        self.combine_fn = get_combine_fn(model_cfg["ensemble_combine_fn"])
        self.properties["num_params"] = self.num_params 

    def to_device(self):
        for model_path in self.models:
            self.models[model_path] = to_device(
                self.models[model_path], self.device, self.config.get("train.channels_last", False)
            )

    def predict(self, x, multi_class, threshold=0.5):
        # Get the label predictions for each model.
        model_outputs = {}
        for model_path in self.models:
            model_outputs[model_path] = self.models[model_path](x)
        # Combine the outputs of the models.
        logit_map = self.combine_fn(model_outputs)
        # Get the hard prediction and probabilities
        prob_map, pred_map = process_logits_map(
            logit_map, 
            multi_class=multi_class, 
            threshold=threshold
            )
        # Return the outputs
        return prob_map, pred_map 