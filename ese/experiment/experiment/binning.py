# local imports
from .utils import process_pred_map, parse_class_name
# torch imports
import torch
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


# Very similar to BaseExperiment, but with a few changes.
class BinningInferenceExp(BaseExperiment):

    def __init__(self, path, set_seed=True, load_data=False):
        # torch.backends.cudnn.benchmark = True
        # super().__init__(path, set_seed)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.build_model()
        # self.build_data(load_data)
        # # Save the config because we've modified it.
        # autosave(self.config.to_dict(), self.path / "config.yml") # Save the new config because we edited it.
        pass
    
    def build_data(self, load_data):
        # # Move the information about channels to the model config.
        # # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        # total_config = self.config.to_dict()
        # # Update the old cfg with new cfg (if it exists).
        # if "data" in self.config:
        #     self.pretrained_data_cfg.update(self.config["data"].to_dict())
        # # Set the data config as the pretrained data config.
        # total_config["data"] = self.pretrained_data_cfg
        # self.config = Config(total_config)
        # # Optionally, load the data.
        # if load_data:
        #     data_cfg = self.config["data"].to_dict()
        #     # Get the dataset class and build the transforms
        #     dataset_cls = absolute_import(data_cfg.pop("_class"))
        #     if "cuda" in data_cfg:
        #         assert data_cfg["preload"], "If you want to cache the dataset on the GPU, you must preload it."
        #         cache_dsets_on_gpu = data_cfg.pop("cuda")
        #     else:
        #         cache_dsets_on_gpu = False
        #     # Build the datasets, apply the transforms
        #     self.train_dataset = dataset_cls(split="val", **data_cfg)
        #     self.val_dataset = dataset_cls(split="cal", **data_cfg)
        #     # Optionally cache the datasets on the GPU.
        #     if cache_dsets_on_gpu:
        #         self.train_dataset = CUDACachedDataset(self.train_dataset)
        #         self.val_dataset = CUDACachedDataset(self.val_dataset)
        pass

    def build_model(self):
        # # Use the total config
        # total_config = self.config.to_dict()
        # # Use the model config.
        # model_cfg = total_config["model"]
        # # Get the configs of the experiment
        # rs = ResultsLoader()
        # # Load ALL the configs in the directory.
        # dfc = rs.load_configs(
        #     model_cfg["pretrained_exp_root"],
        #     properties=False,
        # )
        # # Get the experiment class
        # properties_dir = Path(exp_path) / "properties.json"
        # with open(properties_dir, 'r') as prop_file:
        #     props = json.load(prop_file)
        # exp_class = absolute_import(f'ese.experiment.experiment.{props["experiment"]["class"]}')

        # # Load the experiment
        # loaded_exp = exp_class(exp_path, load_data=False)
        # if model_cfg["checkpoint"] is not None:
        #     loaded_exp.load(tag=model_cfg["checkpoint"])
        # loaded_exp.model.eval()

        # # Set the pretrained data config from the first model.
        # self.pretrained_data_cfg = loaded_exp.config["data"].to_dict()
        # # Figure out what the model and potentially the pretrained backbone are.
        # if hasattr(loaded_exp, "base_model"):
        #     # Do some bookkeping about the kinds of models we are including in the ensemble.
        #     main_model_name = parse_class_name(str(loaded_exp.model.__class__))
        #     pretrained_model_name = parse_class_name(str(loaded_exp.base_model.__class__))
        #     # If the pretrained model_name is identity, then we need to flip the names.
        #     if pretrained_model_name.split('.')[-1] == "Identity":
        #         temp = main_model_name
        #         main_model_name = pretrained_model_name
        #         pretrained_model_name = temp 
        # else:
        #     main_model_name = parse_class_name(str(loaded_exp.model.__class__))
        #     pretrained_model_name = None
        # # Add the model class to the config.
        # model_cfg["_class"] = main_model_name
        # model_cfg["_pretrained_class"] = pretrained_model_name
        # ####################################################
        # # Add other auxilliary information to the config.
        # ####################################################
        # # Count the number of parameters.
        # self.num_params = loaded_exp.properties["num_params"]
        # self.properties["num_params"] = self.num_params 
        # # Create the new config.
        # total_config["model"] = model_cfg
        # self.config = Config(total_config)
        pass

    def predict(self, 
                x, 
                multi_class,
                threshold=0.5,
                return_logits=False):
        # assert x.shape[0] == 1, "Batch size must be 1 for prediction for now."
        # # Predict with the base model.
        # with torch.no_grad():
        #     yhat = self.base_model(x)
        # # Apply post-hoc calibration.
        # if self.model_class == "Vanilla":
        #     yhat_cal = self.model(yhat)
        # else:
        #     yhat_cal = self.model(yhat, image=x)
        # # Get the hard prediction and probabilities
        # prob_map, pred_map = process_pred_map(
        #     yhat_cal, 
        #     multi_class=multi_class, 
        #     threshold=threshold,
        #     return_logits=return_logits
        #     )
        # # Return the outputs
        # return {
        #     'y_pred': prob_map, 
        #     'y_hard': pred_map 
        # }
        pass

    def to_device(self):
        # self.base_model = to_device(self.base_model, self.device, channels_last=False)
        # self.model = to_device(self.model, self.device, channels_last=False)
        pass
