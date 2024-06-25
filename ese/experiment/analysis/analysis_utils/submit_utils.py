# misc imports
import yaml
import random
import itertools
import numpy as np
from pathlib import Path
from typing import List, Optional
from itertools import chain, combinations
# ESE imports
from ese.scripts.utils import gather_exp_paths
from ese.experiment.models.utils import get_calibrator_cls
# Ionpy imports
from ionpy.util import Config
from ionpy.util import dict_product
from ionpy.util.config import check_missing, HDict, valmap


def power_set(in_set):
    return list(chain.from_iterable(combinations(in_set, r) for r in range(len(in_set)+1)))

def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def get_ese_inference_configs(
    exp_cfg: dict,
    base_cfg: Config,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    code_root: Path = Path("/storage/vbutoi/projects/ESE"),
    inf_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/experiment/configs/inference"),
    power_set_keys: Optional[List[str]] = None
):
    # We need to flatten the experiment config to get the different options.
    cfg = HDict(exp_cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    inference_datasets = flat_exp_cfg.pop('data._class')

    # Building new yamls under the exp_group name for model type.
    exp_group = exp_cfg.pop('name')
    model_type = exp_cfg.get('model_type', 'standard')
    base_models = exp_cfg.pop('base_models')
    if 'calibrated_models' in exp_cfg:
        calibrated_models = exp_cfg.pop('calibrated_models')

    # Load the inference cfg from local.
    ##################################################
    default_cfg_root = code_root / "ese" / "experiment" / "configs" / "defaults"
    ##################################################
    with open(default_cfg_root / "Calibration_Metrics.yaml", 'r') as file:
        cal_metrics_cfg = yaml.safe_load(file)
    ##################################################
    base_cfg = base_cfg.update([cal_metrics_cfg])

    # for each power set key, we replace the list of options with its power set.
    if power_set_keys is not None:
        for key in power_set_keys:
            if key in flat_exp_cfg:
                flat_exp_cfg[key] = power_set(flat_exp_cfg[key])

    # Gather the different config options.
    cfg_opt_keys = list(flat_exp_cfg.keys())
    # Generate product tuples by first going through and making sure each option is a list and then using itertools.product.
    for ico_key in flat_exp_cfg:
        if not isinstance(flat_exp_cfg[ico_key], list):
            flat_exp_cfg[ico_key] = [flat_exp_cfg[ico_key]]
    product_tuples = list(itertools.product(*[flat_exp_cfg[key] for key in cfg_opt_keys]))

    # Convert product tuples to dictionaries
    total_run_cfg_options = [{cfg_opt_keys[i]: [item[i]] for i in range(len(cfg_opt_keys))} for item in product_tuples]

    # Keep a list of all the run configuration options.
    inference_opt_list = []

    # If datasets is not a list, make it a list.
    if not isinstance(inference_datasets, list):
        inference_datasets = [inference_datasets]
    inference_datasets = [ifd.split(".")[-1] for ifd in inference_datasets]

    # Define the set of default config options.
    inference_exp_root = scratch_root / "inference" / exp_group
    default_config_options = {
        'experiment.exp_name': [exp_group],
        'experiment.exp_root': [str(inference_exp_root)],
    }

    # Using itertools, get the different combos of calibrators_list ens_cfg_options and ens_w_metric_list.
    for dataset in inference_datasets:

        # Add the dataset specific details.
        with open(inf_cfg_root / f"{dataset}.yaml", 'r') as file:
            dataset_inference_cfg = yaml.safe_load(file)
        # Update the base config with the dataset specific config.
        dataset_base_cfg = base_cfg.update([dataset_inference_cfg])
        # Accumulate a set of config options for each dataset
        dataset_cfgs = []

        for run_opt_dict in total_run_cfg_options: 
            # If you want to run inference on ensembles, use this.
            print(run_opt_dict)
            raise ValueError
            run_opt_args = {
                'experiment.dataset_name': [dataset],
                'log.root': [str(inference_exp_root)],
                'model.pretrained_exp_root': gather_exp_paths(str(model_group_dir)), # Note this is a list of train exp paths.
                **run_opt_dict,
                **default_config_options
            }
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


def get_ese_calibration_configs(
    group_dict: dict,
    calibrators: List[str], 
    cal_base_cfgs: dict,
    cal_model_opts: dict,
    additional_args: Optional[dict] = None,
    subsplit_dict: Optional[dict] = None
):
    scratch_root = Path("/storage/vbutoi/scratch/ESE")
    training_exps_dir = scratch_root / "training" / group_dict['base_models_group']

    cal_option_list = []
    for calibrator in calibrators:
        log_root = scratch_root / 'calibration' / group_dict['exp_group'] / f"Individual_{calibrator}"
        for pt_dir in gather_exp_paths(training_exps_dir):
            # Get the calibrator name
            calibration_options = {
                'log.root': [str(log_root)],
                'train.pretrained_dir': [pt_dir],
            }
            for model_key in cal_base_cfgs[calibrator]:
                if (calibrator in cal_model_opts) and (model_key in cal_model_opts[calibrator]):
                    assert isinstance(cal_model_opts[calibrator][model_key], list), "Calibration model options must be a list."
                    calibration_options[f"model.{model_key}"] = cal_model_opts[calibrator][model_key]
                else:
                    base_cal_val = cal_base_cfgs[calibrator][model_key]
                    assert base_cal_val != "?", "Base calibration model value is not set."
                    calibration_options[f"model.{model_key}"] = [cal_base_cfgs[calibrator][model_key]]

            if subsplit_dict is not None:
                pt_dir_id = pt_dir.split('/')[-1]
                calibration_options['data.subsplit'] = [subsplit_dict[pt_dir_id]]
            if additional_args is not None:
                calibration_options.update(additional_args)
            # Add the calibration options to the list
            cal_option_list.append(calibration_options)
    # Return the list of calibration options
    return cal_option_list