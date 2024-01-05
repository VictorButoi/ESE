# local imports
from .utils import process_logits_map
from .ese_exp import CalibrationExperiment
from ..models.ensemble import get_combine_fn
# torch imports
import torch
# IonPy imports
from ionpy.nn.util import num_params
from ionpy.analysis import ResultsLoader
from ionpy.util.torchutils import to_device
from ionpy.experiment import BaseExperiment 
# misc imports
from typing import Optional


class EnsembleExperiment(BaseExperiment):

    def __init__(
            self, 
            expgroup_path: str, 
            combine_fn: str,
            checkpoint: Optional[str] = None
            ):
        self.build_model(expgroup_path, combine_fn, checkpoint)
        super().__init__(self.paths[0]) # Use the first path as the path for the ensemble.
        self.properties["num_params"] = self.num_params 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def build_model(
            self, 
            expgroup_path: str, 
            combine_fn: str, 
            checkpoint: Optional[str] = None
            ):
        # Get the configs of the experiment
        rs = ResultsLoader()
        # Load ALL the configs in the directory.
        dfc = rs.load_configs(
            expgroup_path,
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
            loaded_exp = CalibrationExperiment(exp_path, build_data=False)
            if checkpoint is not None:
                loaded_exp.load(tag=checkpoint)
            loaded_exp.model.eval()
            self.paths.append(exp_path)
            self.models[exp_path] = loaded_exp.model
            total_params += num_params(loaded_exp.model)
        self.num_params = total_params
        # Build the combine function.
        self.combine_fn = get_combine_fn(combine_fn)

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