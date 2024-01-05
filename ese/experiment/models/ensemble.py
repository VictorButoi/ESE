# local imports
from ..experiment.ese_exp import CalibrationExperiment
# torch imports
import torch
from torch import nn
# IonPy imports
from ionpy.nn.util import num_params
from ionpy.analysis import ResultsLoader


class ModelEnsemble(nn.Module):

    def __init__(self, exp_group_path):
        torch.backends.cudnn.benchmark = True
        super(ModelEnsemble).__init__()
        # Get the configs of the experiment
        rs = ResultsLoader()
        # Load ALL the configs in the directory.
        dfc = rs.load_configs(
            exp_group_path,
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
        self.models= {}
        for exp_path in dfc["path"].unique():
            loaded_exp = CalibrationExperiment(exp_path, build_data=False)
            if self.checkpoint is not None:
                loaded_exp.load(tag=self.checkpoint)
            loaded_exp.model.eval()
            self.models[exp_path] = loaded_exp.model

    def forward(self, x):
        # Get the label predictions for each model.
        model_outputs = {}
        for model_path in self.models:
            model_outputs[model_path] = self.models[model_path](x)
        # Combine the outputs of the models.
        output = self.combine_fn(model_outputs)
        # Return the outputs
        return output
