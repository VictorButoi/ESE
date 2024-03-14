# local imports
from .binning_exp import BinningInferenceExperiment
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
from ionpy.experiment.util import absolute_import
# misc imports
import json
from pathlib import Path
from typing import Optional, Literal


# Very similar to BaseExperiment, but with a few changes.
class EnsembleInferenceExperiment(BaseExperiment):

    def __init__(self, path, set_seed=True):
        torch.backends.cudnn.benchmark = True
        super().__init__(path, set_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.build_data()
        # Save the config because we've modified it.
        autosave(self.config.to_dict(), self.path / "config.yml") # Save the new config because we edited it.

    def build_model(self):
        # Use the total config
        total_config = self.config.to_dict()
        # Use the model config.
        model_cfg = total_config["model"]
        ensemble_cfg = total_config["ensemble"]
        # Get the configs of the experiment
        rs = ResultsLoader()
        # Load ALL the configs in the directory.
        if "pretrained_exp_root" in model_cfg and model_cfg["pretrained_exp_root"] != "None":
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
                    if column not in ["seed", "subsplit", "path", "pretrained_dir"] and\
                        len(dfc[column].unique()) > 1:
                        raise ValueError(f"The only difference between the configs should be the seed, "\
                                        + f"but found different values in column '{column}'."\
                                            + f"Unique values: {dfc[column].unique()}")
            # Verify that the configs are valid.
            verify_ensemble_configs(dfc)
            # Gather the unique ensemble paths.
            member_exp_path_list = dfc["path"].unique()
        elif "member_paths" in ensemble_cfg:
            member_exp_path_list = ensemble_cfg["member_paths"]
        else:
            raise ValueError("No pretrained ensemble root or member paths found in config.")

        # Loop through each config and build the experiment, placing it in a dictionary.
        self.normalize = ensemble_cfg["normalize"]
        self.combine_fn = ensemble_cfg["combine_fn"]
        self.combine_quantity = ensemble_cfg["combine_quantity"]
        self.ens_exp_paths = []
        self.ens_exp_seeds = []

        self.ens_exps = {}
        self.num_params = 0
        # Loop through each member of the ensemble.
        for exp_idx, exp_path in enumerate(member_exp_path_list):
            # Special case where we want to use the binning calibrator.
            if "Binning" in model_cfg["calibrator"]:  
                # Construct the cfg for this binning member
                binning_cfg = {
                    "log": {
                        "root": total_config["log"]["root"] + "/binning_exp_logs"
                    },
                    "global_calibration": total_config["global_calibration"],
                    "experiment": total_config["experiment"],
                    "model": {
                        "pretrained_exp_root": str(exp_path),
                        "calibrator_cls": model_cfg["calibrator_cls"],
                        "cal_stats_split": model_cfg["cal_stats_split"],
                        "normalize": model_cfg["normalize"],
                    },
                    "data": total_config["data"]
                }
                # Load the experiment
                loaded_exp = BinningInferenceExperiment.from_config(binning_cfg)
            # Otheriwse, get the experiment class corresponding to the exp_path.
            else:
                properties_dir = Path(exp_path) / "properties.json"
                with open(properties_dir, 'r') as prop_file:
                    props = json.load(prop_file)
                exp_class = absolute_import(f'ese.experiment.experiment.{props["experiment"]["class"]}')
                # Load the experiment
                loaded_exp = exp_class(exp_path, load_data=False)
                # Load the experiment weights.
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
        # Sort the ensemble paths by their seeds.
        self.ens_exp_paths = [x for _, x in sorted(zip(self.ens_exp_seeds, self.ens_exp_paths))]

        ####################################################
        # Add other auxilliary information to the config.
        ####################################################
        # Count the number of parameters.
        self.properties["num_params"] = self.num_params 
        # Create the new config.
        total_config["model"] = model_cfg
        self.config = Config(total_config)
    
    def build_data(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        # Update the old cfg with new cfg (if it exists).
        if "data" in self.config:
            self.pretrained_data_cfg.update(self.config["data"].to_dict())
        # Set the data config as the pretrained data config.
        total_config["data"] = self.pretrained_data_cfg
        self.config = Config(total_config)

    def to_device(self):
        for exp_path in self.ens_exp_paths:
            self.ens_exps[exp_path].to_device()

    def predict(
        self, 
        x: torch.Tensor, 
        multi_class: bool, 
        threshold: float = 0.5, 
        normalize: Optional[bool] = None,
        combine_fn: Optional[str] = None,
        combine_quantity: Optional[Literal["probs", "logits"]] = None
    ):
        # Get the label predictions for each model.
        ensemble_model_outputs = []
        return_logits = None
        for exp_path in self.ens_exp_paths:
            # Multi-class needs to be true here so that we can combine the outputs.
            member_pred = self.ens_exps[exp_path].predict(
                x=x, 
                multi_class=True
            )
            if 'y_logits' in member_pred:
                return_logits = True
                ensemble_model_outputs.append(member_pred['y_logits'])
            else:
                return_logits = False
                ensemble_model_outputs.append(member_pred['y_probs'])

        # Combine the outputs of the models.
        if combine_fn is None:
            assert self.combine_fn is not None, "No combine function provided."
            combine_fn = self.combine_fn
        if combine_quantity is None:
            assert self.combine_quantity is not None, "No pre_softmax value provided."
            combine_quantity = self.combine_quantity
        if normalize is None:
            assert self.normalize is not None, "No normalization value provided."
            normalize = self.normalize

        # Combine the outputs of the models.
        combined_outputs = get_combine_fn(combine_fn)(
            ensemble_model_outputs, 
            combine_quantity=combine_quantity,
            normalize=normalize,
            from_logits=return_logits 
        )
        # Get the hard prediction and probabilities, if we are doing identity,
        # then we don't want to return probs.
        if combine_fn == "identity":
            if return_logits:
                return {
                    'y_logits': combined_outputs,
                }
            else:
                return {
                    'y_probs': combined_outputs,
                }
        else:
            prob_map, pred_map = process_pred_map(
                combined_outputs, 
                multi_class=multi_class, 
                threshold=threshold,
                from_logits=return_logits
            )
            # Return the outputs
            return {
                'y_probs': prob_map, 
                'y_hard': pred_map # if identity, this will be None.
            }