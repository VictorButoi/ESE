# misc imports
import os
import yaml
import itertools
from pathlib import Path
from pprint import pprint
from datetime import datetime
from typing import List, Optional
from pydantic import validate_arguments
from itertools import chain, combinations
# ESE imports
from ese.scripts.utils import gather_exp_paths, get_option_product
# Ionpy imports
from ionpy.util import Config
from ionpy.util import dict_product
from ionpy.util.ioutil import autosave
from ionpy.util.config import check_missing, HDict, valmap


def power_set(in_set):
    return list(chain.from_iterable(combinations(in_set, r) for r in range(len(in_set)+1)))

def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ese_training_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    train_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/experiment/configs/training"),
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    train_exp_root = scratch_root / "training" / exp_cfg['name']
    autosave(exp_cfg, train_exp_root / "experiment.yml") # SAVE #1: Experiment config
    exp_name = exp_cfg.pop('name')

    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        exp_name = f"{formatted_date}_{exp_name}"

    cfg = HDict(exp_cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    train_dataset_name = flat_exp_cfg['data._class'].split('.')[-1]

    # Load the dataset specific config and update the base config.
    with open(train_cfg_root/ f"{train_dataset_name}.yaml", 'r') as file:
        dataset_cfg = yaml.safe_load(file)
    base_cfg = base_cfg.update([dataset_cfg])
    autosave(base_cfg.to_dict(), train_exp_root / "base.yml") # SAVE #2: Base config
    
    # Get the information about seeds.
    seed = flat_exp_cfg.pop('experiment.seed', 40)
    seed_range = flat_exp_cfg.pop('experiment.seed_range', 1)
    
    # We need all of our options to be in lists as convention for the product.
    for ico_key in flat_exp_cfg:
        if not isinstance(flat_exp_cfg[ico_key], list):
            flat_exp_cfg[ico_key] = [flat_exp_cfg[ico_key]]
    
    # Create the ablation options.
    option_set = {
        'log.root': [str(train_exp_root)],
        'experiment.seed': [seed + seed_idx for seed_idx in range(seed_range)],
        **flat_exp_cfg
    }

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg)

    # Return the train configs.
    return cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ese_calibration_configs(
    exp_cfg: dict,
    base_cfg: Config,
    calibration_model_cfgs: dict,
    add_date: bool = True,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    calibration_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/experiment/configs/calibrate"),
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    calibration_exp_root = scratch_root / "calibration" / exp_cfg['name']
    autosave(exp_cfg, calibration_exp_root / "experiment.yml") # SAVE #1: Experiment config
    exp_name = exp_cfg.pop('name')

    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        exp_name = f"{formatted_date}_{exp_name}"
    
    cfg = HDict(exp_cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    calibration_dataset_name = flat_exp_cfg['data._class'].split('.')[-1]

    # Load the dataset specific config and update the base config.
    with open(calibration_cfg_root/ f"{calibration_dataset_name}.yaml", 'r') as file:
        dataset_cfg = yaml.safe_load(file)
    base_cfg = base_cfg.update([dataset_cfg])
    autosave(base_cfg.to_dict(), calibration_exp_root / "base.yml") # SAVE #2: Base config
    
    # We need all of our options to be in lists as convention for the product.
    for ico_key in flat_exp_cfg:
        # If this is a tuple, then convert it to a list.
        if isinstance(flat_exp_cfg[ico_key], tuple):
            flat_exp_cfg[ico_key] = list(flat_exp_cfg[ico_key])
        # Otherwise, make sure it is a list.
        elif not isinstance(flat_exp_cfg[ico_key], list):
            flat_exp_cfg[ico_key] = [flat_exp_cfg[ico_key]]
    
    # Create the ablation options.
    option_set = {
        'log.root': [str(calibration_exp_root)],
        **flat_exp_cfg
    }

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg)

    # This is a list of calibration model configs. But the actual calibration model
    # should still not be defined at this point. We iterate through the configs, and replace
    # the model config with the calibration model config.
    for c_idx, cfg in enumerate(cfgs):
        # Convert the Config obj to a dict.
        cfg_dict = cfg.to_dict()
        # Replace the model with the dict from calibration model cfgs.
        cal_model = cfg_dict.pop('model')
        cfg_dict['model'] = calibration_model_cfgs[cal_model]
        # Replace the Config object with the new config dict.
        cfgs[c_idx] = Config(cfg_dict)

    # Return the train configs.
    return cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ese_inference_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    power_set_keys: Optional[List[str]] = None,
    code_root: Path = Path("/storage/vbutoi/projects/ESE"),
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    inf_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/experiment/configs/inference"),
):
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    # Save the experiment config.
    inference_exp_root = scratch_root / "inference" / exp_cfg['name']
    autosave(exp_cfg, inference_exp_root / "experiment.yml") # SAVE #1: Experiment config
    exp_name = exp_cfg.pop('name')

    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        exp_name = f"{formatted_date}_{exp_name}"

    cfg = HDict(exp_cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    inference_datasets = flat_exp_cfg.pop('data._class')
    # For any key that is a tuple we need to convert it to a list, this is an artifact of the flattening..
    for key in flat_exp_cfg:
        if isinstance(flat_exp_cfg[key], tuple):
            flat_exp_cfg[key] = list(flat_exp_cfg[key])

    # Load the inference cfg from local.
    ##################################################
    default_cfg_root = code_root / "ese" / "experiment" / "configs" / "defaults"
    ##################################################
    with open(default_cfg_root / "Calibration_Metrics.yaml", 'r') as file:
        cal_metrics_cfg = yaml.safe_load(file)
    ##################################################
    base_cfg = base_cfg.update([cal_metrics_cfg])
    autosave(base_cfg.to_dict(), inference_exp_root / "base.yml") # SAVE #2: Base config

    # for each power set key, we replace the list of options with its power set.
    if power_set_keys is not None:
        for key in power_set_keys:
            if key in flat_exp_cfg:
                flat_exp_cfg[key] = power_set(flat_exp_cfg[key])

    # Gather the different config options.
    cfg_opt_keys = list(flat_exp_cfg.keys())

    #First going through and making sure each option is a list and then using itertools.product.
    for ico_key in flat_exp_cfg:
        if not isinstance(flat_exp_cfg[ico_key], list):
            flat_exp_cfg[ico_key] = [flat_exp_cfg[ico_key]]
    
    # Generate product tuples 
    product_tuples = list(itertools.product(*[flat_exp_cfg[key] for key in cfg_opt_keys]))

    # Convert product tuples to dictionaries
    total_run_cfg_options = [{cfg_opt_keys[i]: [item[i]] for i in range(len(cfg_opt_keys))} for item in product_tuples]
    # Keep a list of all the run configuration options.
    inference_opt_list = []

    # If datasets is not a list, make it a list.
    if not isinstance(inference_datasets, list):
        inference_datasets = [inference_datasets]
    inf_dataset_names = [ifd.split(".")[-1] for ifd in inference_datasets]

    # Define the set of default config options.
    default_config_options = {
        'experiment.exp_name': [exp_name],
        'experiment.exp_root': [str(inference_exp_root)],
    }

    # Using itertools, get the different combos of calibrators_list ens_cfg_options and ens_w_metric_list.
    for d_idx, dataset_name in enumerate(inf_dataset_names):
        # Add the dataset specific details.
        with open(inf_cfg_root / f"{dataset_name}.yaml", 'r') as file:
            dataset_inference_cfg = yaml.safe_load(file)
        # Update the base config with the dataset specific config.
        dataset_base_cfg = base_cfg.update([dataset_inference_cfg])
        # Accumulate a set of config options for each dataset
        dataset_cfgs = []
        # Iterate through all of our inference options.
        for run_opt_dict in total_run_cfg_options: 
            # One required key is 'base_model'. We need to know if it is a single model or a group of models.
            # We evaluate this by seeing if 'submitit' is in the base model path.
            model_group_dir = Path(run_opt_dict.pop('base_model')[0])
            # If you want to run inference on a single model, use this.
            run_opt_args = {
                'log.root': [str(inference_exp_root)],
                'data._class': [inference_datasets[d_idx]],
                'experiment.dataset_name': [dataset_name],
                **run_opt_dict,
                **default_config_options
            }
            if 'submitit' in os.listdir(model_group_dir):
                run_opt_args['model.pretrained_exp_root'] = gather_exp_paths(str(model_group_dir)) 
            else:
                run_opt_args['model.pretrained_exp_root'] = [str(model_group_dir)]
            # Append these to the list of configs and roots.
            dataset_cfgs.append(run_opt_args)
        # Iterate over the different config options for this dataset. 
        for option_dict in dataset_cfgs:
            for cfg_update in dict_product(option_dict):
                cfg = dataset_base_cfg.update([cfg_update])
                # Verify it's a valid config
                check_missing(cfg)
                # Add it to the total list of inference options.
                inference_opt_list.append(cfg)

    # Return the list of different configs.
    return inference_opt_list
