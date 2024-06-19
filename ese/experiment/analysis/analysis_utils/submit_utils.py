# misc imports
import yaml
import random
import itertools
import numpy as np
from pathlib import Path
from typing import Any, List, Optional
from itertools import chain, combinations
# ESE imports
from ese.scripts.utils import gather_exp_paths
from ese.experiment.models.utils import get_calibrator_cls
# Ionpy imports
from ionpy.util import Config
from ionpy.util import dict_product
from ionpy.util.config import check_missing


def power_set(in_set):
    return list(chain.from_iterable(combinations(in_set, r) for r in range(len(in_set)+1)))


def get_ese_inference_configs(
    experiment_cfg: dict,
    base_cfg: Config,
    code_root: str = "/storage/vbutoi/projects/ESE",
    scratch_root: str = "/storage/vbutoi/scratch/ESE",
    inf_cfg_root: str = "/storage/vbutoi/projects/ESE/ese/experiment/configs/inference",
    power_set_keys: Optional[List[str]] = None
):
    # Load the inference cfg from local.
    ##################################################
    inf_cfg_root = Path(code_root) / "ese" / "experiment" / "configs" / "defaults"

    ##################################################
    with open(inf_cfg_root / "Calibration_Metrics.yaml", 'r') as file:
        cal_metrics_cfg = yaml.safe_load(file)

    ##################################################
    base_cfg = base_cfg.update([cal_metrics_cfg])

    # for each power set key, we replace the list of options with its power set.
    if power_set_keys is not None:
        for key in power_set_keys:
            if key in inf_cfg_opts:
                inf_cfg_opts[key] = power_set(inf_cfg_opts[key])

    # Gather the different config options.
    cfg_opt_keys = list(inf_cfg_opts.keys())
    if 'calibrator' in cfg_opt_keys:
        cfg_opt_keys.remove('calibrator') # We need to handle calibrator separately.
    else:
        inf_cfg_opts['calibrator'] = ['Uncalibrated']

    # Generate product tuples by first going through and making sure each option is a list and then using itertools.product.
    for ico_key in inf_cfg_opts:
        if not isinstance(inf_cfg_opts[ico_key], list):
            inf_cfg_opts[ico_key] = [inf_cfg_opts[ico_key]]
    product_tuples = list(itertools.product(*[inf_cfg_opts[key] for key in cfg_opt_keys]))
    
    # Convert product tuples to dictionaries
    total_run_cfg_options = [{cfg_opt_keys[i]: [item[i]] for i in range(len(cfg_opt_keys))} for item in product_tuples]

    # Keep a list of all the run configuration options.
    inference_opt_list = []
    # If datasets is not a list, make it a list.
    if not isinstance(inference_datasets, list):
        inference_datasets = [inference_datasets]
    # Using itertools, get the different combos of calibrators_list ens_cfg_options and ens_w_metric_list.
    for dataset in inference_datasets:
        # Accumulate a set of config options for each dataset
        dataset_cfgs = []
        for calibrator in inf_cfg_opts['calibrator']:
            ##################################################
            # Set a few things that will be consistent for all runs.
            ##################################################
            inference_exp_root = Path(f"{scratch_root}/inference/{exp_group}")

            # Define the set of default config options.
            default_config_options = {
                'experiment.exp_name': [exp_group],
                'experiment.exp_root': [str(inference_exp_root)],
                'experiment.dataset_name': [dataset],
                'calibrator._name': [calibrator],
                'calibrator._class': [get_calibrator_cls(calibrator)],
            }

            # Define where we get the base models from.
            use_uncalibrated_models = (calibrator == "Uncalibrated") or ("Binning" in calibrator)
            if use_uncalibrated_models:
                model_group_dir = base_models_group
                default_config_options['model.checkpoint'] = ['max-val-dice_score']
            else:
                model_group_dir = f"{calibrated_models_group}/Individual_{calibrator}"
                default_config_options['model.checkpoint'] = ['min-val-ece_loss']

            #####################################
            # Choose the ensembles ahead of time.
            #####################################
            if np.any([run_opt_dict.get('do_ensemble', False) for run_opt_dict in total_run_cfg_options]) and model_type != "incontext":
                # Get all unique subsets of total_ens_members of size num_+ens_members.
                ensemble_group = list(itertools.combinations(gather_exp_paths(str(model_group_dir)), base_cfg['ensemble']['num_members']))
                # We need to subsample the unique ensembles or else we will be here forever.
                max_ensemble_samples = base_cfg['experiment']['max_ensemble_samples']
                if len(ensemble_group) > max_ensemble_samples:
                    ensemble_group = random.sample(ensemble_group, k=max_ensemble_samples)

            for run_opt_dict in total_run_cfg_options: 
                # If you want to run inference on ensembles, use this.
                if run_opt_dict.get('model.ensemble', False) or model_type == "incontext":
                    # Define where we want to save the results.
                    ensemble_log_folder = f"{dataset}_Ensemble_{calibrator}"
                    # Define where the set of base models come from.
                    advanced_args = {
                        "log.root": [str(inference_exp_root / ensemble_log_folder)],
                        "model.ensemble": [True],
                        **run_opt_dict
                    }
                    ensemble_cfg_args = {
                        'ensemble.num_members': [1]
                        **default_config_options,
                        **advanced_args
                    }
                    # For each num_ens_members, we subselect that num of the total_ens_members.
                    if model_type == "incontext":
                        ensemble_cfg_args['model.pretrained_exp_root'] = [str(model_group_dir)]
                    else:
                        ensemble_cfg_args['ensemble.member_paths'] = [list(ensemble_group)]
                    # Append these to the list of configs and roots.
                    dataset_cfgs.append(ensemble_cfg_args)
                # If you want to run inference on individual networks, use this.
                else:
                    # If we aren't ensembling, then we can't do incontext models.
                    assert model_type != "incontext", "Incontext models can only be used with ensembles."
                    run_opt_args = {
                        'log.root': [str(inference_exp_root / f"{dataset}_Individual_{calibrator}")],
                        'model.pretrained_exp_root': gather_exp_paths(str(model_group_dir)), # Note this is a list of train exp paths.
                        **run_opt_dict,
                        **default_config_options
                    }
                    # Append these to the list of configs and roots.
                    dataset_cfgs.append(run_opt_args)
            
        # Finally, add the dataset specific details.
        with open(f"{inf_cfg_root}/{dataset}.yaml", 'r') as file:
            dataset_inference_cfg = yaml.safe_load(file)
        # Update the base config with the dataset specific config.
        dataset_base_cfg = base_cfg.update([dataset_inference_cfg])
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