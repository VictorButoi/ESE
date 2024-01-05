# local imports
from .ese_exp import CalibrationExperiment
# torch imports
import torch
# IonPy imports
from ionpy.experiment.util import fix_seed
from ionpy.experiment import BaseExperiment 
from ionpy.nn.util import num_params
from ionpy.analysis import ResultsLoader


class EnsembleExperiment(BaseExperiment):

    def __init__(
            self, 
            expgroup_path, 
            checkpoint=None, 
            seed=None
            ):
        self.build_model(expgroup_path, checkpoint)
        super().__init__(self.paths[0]) # Use the first path as the path for the ensemble.
        self.properties["num_params"] = self.num_params 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Make sure that we are not changing the seed.
        if seed is not None:
            fix_seed(seed)
        
    def build_model(self, expgroup_path, checkpoint=None):
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
    
    def predict(self, x):
        # Get the label predictions for each model.
        model_outputs = {}
        for model_path in self.models:
            model_outputs[model_path] = self.models[model_path](x)
        # Combine the outputs of the models.
        output = self.combine_fn(model_outputs)
        # Return the outputs
        return output