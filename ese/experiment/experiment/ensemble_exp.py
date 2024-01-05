# local imports
from .ese_exp import CalibrationExperiment
# torch imports
import torch
from torch.utils.data import DataLoader
# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.datasets.cuda import CUDACachedDataset
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.nn.util import num_params
from ionpy.util import Config
from ionpy.util.torchutils import to_device
from ionpy.analysis import ResultsLoader


class EnsembleExperiment(TrainExperiment):

    def __init__(self, expgroup_path, checkpoint="max-val-dice_score"):
        self.build_model(expgroup_path, checkpoint)

    def build_data(self):
        # Get the data and transforms we want to apply
        pretrained_data_cfg = self.pretrained_exp.config["data"].to_dict()
        # Update the old cfg with new cfg (if it exists).
        if "data" in self.config:
            pretrained_data_cfg.update(self.config["data"].to_dict())
        # Get the dataset class and build the transforms
        dataset_cls = absolute_import(pretrained_data_cfg.pop("_class"))
        if "cuda" in pretrained_data_cfg:
            assert pretrained_data_cfg["preload"], "If you want to cache the dataset on the GPU, you must preload it."
            cache_dsets_on_gpu = pretrained_data_cfg.pop("cuda")
        else:
            cache_dsets_on_gpu = False
        # Build the datasets, apply the transforms
        self.train_dataset = dataset_cls(split="val", **pretrained_data_cfg)
        self.val_dataset = dataset_cls(split="cal", **pretrained_data_cfg)
        # Optionally cache the datasets on the GPU.
        if cache_dsets_on_gpu:
            self.train_dataset = CUDACachedDataset(self.train_dataset)
            self.val_dataset = CUDACachedDataset(self.val_dataset)

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
        self.models= {}
        total_params = 0
        for exp_path in dfc["path"].unique():
            loaded_exp = CalibrationExperiment(exp_path, build_data=False)
            if checkpoint is not None:
                loaded_exp.load(tag=checkpoint)
            loaded_exp.model.eval()
            self.models[exp_path] = loaded_exp.model
            total_params += num_params(loaded_exp.model)
        self.properties["num_params"] = num_params
    
    def predict(self, x):
        # Get the label predictions for each model.
        model_outputs = {}
        for model_path in self.models:
            model_outputs[model_path] = self.models[model_path](x)
        # Combine the outputs of the models.
        output = self.combine_fn(model_outputs)
        # Return the outputs
        return output