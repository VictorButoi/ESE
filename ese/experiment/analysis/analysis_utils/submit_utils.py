# misc imports
import ast
import yaml
import random
import itertools
import numpy as np
from pathlib import Path
from typing import List, Optional
# ESE imports
from ese.scripts.utils import gather_exp_paths
from ese.experiment.models.utils import get_calibrator_cls
# Ionpy imports
from ionpy.util import Config
from ionpy.util import dict_product
from ionpy.experiment.util import fix_seed
from ionpy.util.config import check_missing

def get_ese_inference_configs(
    group_dict: dict,
    inf_cfg_opts: dict,
    base_cfg: Config,
    base_cfg_args: dict
):
    # Set the seed if it is provided.
    if "seed" in base_cfg_args['submit_opts']:
        fix_seed(base_cfg_args['submit_opts']['seed'])

    # Gather the different config options.
    keys = list(inf_cfg_opts.keys())
    keys.remove('calibrator') # We need to handle calibrator separately.
    # Generate product tuples
    product_tuples = list(itertools.product(*[inf_cfg_opts[key] for key in keys]))
    # Convert product tuples to dictionaries
    total_run_cfg_options = [{keys[i]: [item[i]] for i in range(len(keys))} for item in product_tuples]

    # Keep a list of all the run configuration options.
    inference_opt_list = []
    # Using itertools, get the different combos of calibrators_list ens_cfg_options and ens_w_metric_list.
    for dataset in group_dict['datasets']:

        # Accumulate a set of config options for each dataset
        dataset_cfgs = []
        
        for calibrator in inf_cfg_opts['calibrator']:
            ##################################################
            # Set a few things that will be consistent for all runs.
            ##################################################
            inference_exp_root = Path(group_dict['scratch_root']) / "inference" / group_dict['exp_group']

            # Define the set of default config options.
            default_config_options = {
                'experiment.exp_root': [str(inference_exp_root)],
                'experiment.dataset_name': [dataset],
                'calibrator._name': [calibrator],
                'calibrator._class': [get_calibrator_cls(calibrator)],
            }
            if 'preload' in group_dict:
                default_config_options['data.preload'] = [group_dict['preload']]
            # If additional args are provided, update the default config options.
            if 'exp_opts' in base_cfg_args:
                default_config_options.update(base_cfg_args['exp_opts'])

            # Define where we get the base models from.
            use_uncalibrated_models = (calibrator == "Uncalibrated") or ("Binning" in calibrator)
            if use_uncalibrated_models:
                model_group_dir = group_dict['base_models_group']
            else:
                model_group_dir = group_dict['calibrated_models_group'] / f"Individual_{calibrator}"

            #####################################
            # Choose the ensembles ahead of time.
            #####################################
            if np.any([run_opt_dict.get('do_ensemble', False) for run_opt_dict in total_run_cfg_options]) and group_dict['model_type'] != "incontext":
                total_ens_members = gather_exp_paths(str(model_group_dir))
                ensemble_groups = {}
                for num_ens_members in base_cfg_args['submit_opts']['num_ens_membs']:
                    # Get all unique subsets of total_ens_members of size num_+ens_members.
                    unique_ensembles = list(itertools.combinations(total_ens_members, num_ens_members))
                    # We need to subsample the unique ensembles or else we will be here forever.
                    if len(unique_ensembles) > base_cfg_args['submit_opts']['max_ensemble_samples']:
                        ensemble_groups[num_ens_members] = random.sample(unique_ensembles, k=base_cfg_args['submit_opts']['max_ensemble_samples'])
                    else:
                        ensemble_groups[num_ens_members] = unique_ensembles

            for run_opt_dict in total_run_cfg_options: 
                # If you want to run inference on ensembles, use this.
                if run_opt_dict.get('model.ensemble', False) or group_dict['model_type'] == "incontext":
                    # For each ensemble option, we want to run inference.
                    for ens_cfg in base_cfg_args['submit_opts']['ens_cfg_options']:
                        # Make the ens_cfg a tuple.
                        ens_cfg = ast.literal_eval(ens_cfg)
                        # Define where we want to save the results.
                        if base_cfg_args['submit_opts'].get('ensemble_upper_bound', False):
                            inf_log_root = inference_exp_root / f"ensemble_upper_bounds"
                        else:
                            inf_log_root = inference_exp_root / f"{dataset}_Ensemble_{calibrator}"
                        # Define where the set of base models come from.
                        advanced_args = {
                            "log.root": [str(inf_log_root)],
                            "model.ensemble": [True],
                            "ensemble.combine_fn": [ens_cfg[0]],
                            "ensemble.combine_quantity": [ens_cfg[1]],
                            **run_opt_dict
                        }
                        for num_ens_members in base_cfg_args['submit_opts']['num_ens_membs']:
                            # For each num_ens_members, we subselect that num of the total_ens_members.
                            if group_dict['model_type'] == "incontext":
                                # Make a copy of our default config options.
                                dupe_def_cfg_opts = default_config_options.copy()
                                # If we are using incontext models, we need to use the ensemble groups.
                                advanced_args['ensemble.num_members'] = [num_ens_members]
                                advanced_args['model.pretrained_exp_root'] = [str(model_group_dir)]
                                advanced_args['experiment.supports_per_target'] = [base_cfg_args['submit_opts']['supports_per_target']]
                                # Combine the default and advanced arguments
                                dupe_def_cfg_opts.update(advanced_args)
                                # Append these to the list of configs and roots.
                                dataset_cfgs.append(dupe_def_cfg_opts)
                            else:
                                for ens_group in ensemble_groups[num_ens_members]:
                                    # Make a copy of our default config options.
                                    dupe_def_cfg_opts = default_config_options.copy()
                                    # Define where the set of base models come from.
                                    advanced_args['ensemble.num_members'] = [num_ens_members]
                                    advanced_args['ensemble.member_paths'] = [list(ens_group)]
                                    # Combine the default and advanced arguments.
                                    dupe_def_cfg_opts.update(advanced_args)
                                    # Append these to the list of configs and roots.
                                    dataset_cfgs.append(dupe_def_cfg_opts)
                # If you want to run inference on individual networks, use this.
                else:
                    # If we aren't ensembling, then we can't do incontext models.
                    assert group_dict['model_type'] != "incontext", "Incontext models can only be used with ensembles."
                    advanced_args = {
                        'log.root': [str(inference_exp_root / f"{dataset}_Individual_{calibrator}")],
                        'model.ensemble': [False],
                        'model.pretrained_exp_root': gather_exp_paths(str(model_group_dir)), # Note this is a list of train exp paths.
                        'ensemble.normalize': [None],
                        'ensemble.combine_fn': [None],
                        'ensemble.combine_quantity': [None],
                        **run_opt_dict
                    }
                    # Combine the default and advanced arguments.
                    default_config_options.update(advanced_args)
                    # Append these to the list of configs and roots.
                    dataset_cfgs.append(default_config_options)
            
        # Finally, add the dataset specific details.
        with open(f"{group_dict['inf_cfg_root']}/datasets/{group_dict['model_type']}/{dataset}.yaml", 'r') as file:
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