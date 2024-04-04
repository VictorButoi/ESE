# misc imports
import ast
import random
import itertools
import numpy as np
from pathlib import Path
from typing import List, Optional
# ESE imports
from ese.scripts.utils import gather_exp_paths
from ese.experiment.models.utils import get_calibrator_cls
# Ionpy imports
from ionpy.experiment.util import fix_seed

def get_ese_inference_configs(
    group_dict: dict,
    inf_cfg_opts: dict,
    base_cfg_args: dict
):
    scratch_root = Path(base_cfg_args['submit_opts']['scratch_root'])

    # Set the seed if it is provided.
    if "seed" in base_cfg_args['submit_opts']:
        fix_seed(base_cfg_args['submit_opts']['seed'])

    # Gather the different config options.
    keys = list(inf_cfg_opts.keys())
    keys.remove('calibrator') # We need to handle calibrator separately.
    # Generate product tuples
    product_tuples = list(itertools.product(*[inf_cfg_opts[key] for key in keys]))
    # Convert product tuples to dictionaries
    total_run_cfg_options = [{keys[i]: item[i] for i in range(len(keys))} for item in product_tuples]

    # Keep a list of all the run configuration options.
    calibrator_option_list = []
    # Using itertools, get the different combos of calibrators_list ens_cfg_options and ens_w_metric_list.
    for calibrator in inf_cfg_opts['calibrator']:
        ##################################################
        # Set a few things that will be consistent for all runs.
        ##################################################
        exp_root = scratch_root / "inference" / group_dict['exp_group']
        use_uncalibrated_models = (calibrator == "Uncalibrated") or ("Binning" in calibrator)
        # Define the set of default config options.
        default_config_options = {
            'experiment.exp_root': [str(exp_root)],
            'experiment.dataset_name': [group_dict['dataset']],
            'model.calibrator': [calibrator],
            'model.calibrator_cls': [get_calibrator_cls(calibrator)],
        }
        if 'preload' in group_dict:
            default_config_options['data.preload'] = [group_dict['preload']]

        # If additional args are provided, update the default config options.
        default_config_options.update(base_cfg_args['exp_opts'])

        # Define where we get the base models from.
        if use_uncalibrated_models:
            inf_group_dir = scratch_root / "training" / group_dict['base_models_group']
        else:
            inf_group_dir = scratch_root / "calibration" / group_dict['calibrated_models_group'] / f"Individual_{calibrator}"

        #####################################
        # Choose the ensembles ahead of time.
        #####################################
        if np.any([run_opt_dict.get('do_ensemble', False) for run_opt_dict in total_run_cfg_options]):
            total_ens_members = gather_exp_paths(str(inf_group_dir))
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
            if run_opt_dict['do_ensemble']:
                # For each ensemble option, we want to run inference.
                for ens_cfg in base_cfg_args['submit_opts']['ens_cfg_options']:
                    # Make the ens_cfg a tuple.
                    ens_cfg = ast.literal_eval(ens_cfg)
                    # Define where we want to save the results.
                    if base_cfg_args['submit_opts'].get('ensemble_upper_bound', False):
                        inf_log_root = exp_root / f"ensemble_upper_bounds"
                    else:
                        inf_log_root = exp_root / f"{group_dict['dataset']}_Ensemble_{calibrator}"
                    # For each num_ens_members, we subselect that num of the total_ens_members.
                    for num_ens_members in base_cfg_args['submit_opts']['num_ens_membs']:
                        for ens_group in ensemble_groups[num_ens_members]:
                            # Make a copy of our default config options.
                            dupe_def_cfg_opts = default_config_options.copy()
                            # Define where the set of base models come from.
                            advanced_args = {
                                'log.root': [str(inf_log_root)],
                                'model.ensemble': [True],
                                'ensemble.combine_fn': [ens_cfg[0]],
                                'ensemble.combine_quantity': [ens_cfg[1]],
                                'ensemble.member_paths': [list(ens_group)],
                            }
                            if 'member_temps' in run_opt_dict:
                                advanced_args['ensemble.member_temps'] = [run_opt_dict['member_temps']]
                            elif 'member_temps_upper_bound' in base_cfg_args['submit_opts']:
                                # Flip a coin to see if we are going to use the upper bound or lower bound, per member.
                                # This equates to randomly sampling num_ens_members many bernoullis.
                                under_vs_over_conf = np.random.binomial(n=1, p=0.5, size=num_ens_members)
                                # Get the overconfident temps.
                                over_conf_temps = np.random.uniform(0.01, 1.0, size=num_ens_members)
                                # Get the underconfident tempts.
                                under_conf_temps = np.random.uniform(1.0, base_cfg_args['submit_opts']['member_temps_upper_bound'], size=num_ens_members)
                                # Build the temps vector accordingly
                                members_temps = [over_conf_temps[i] if under_vs_over_conf[i] else under_conf_temps[i] for i in range(num_ens_members)]
                                advanced_args['ensemble.member_temps'] = [str(tuple(members_temps))]
                            # Combine the default and advanced arguments.
                            dupe_def_cfg_opts.update(advanced_args)
                            # Append these to the list of configs and roots.
                            calibrator_option_list.append(dupe_def_cfg_opts)
            # If you want to run inference on individual networks, use this.
            else:
                advanced_args = {
                    'log.root': [str(exp_root / f"{group_dict['dataset']}_Individual_{calibrator}")],
                    'model.ensemble': [False],
                    'ensemble.normalize': [None],
                    'ensemble.combine_fn': [None],
                    'ensemble.combine_quantity': [None],
                }
                if base_cfg_args['submit_opts']['gather_sub_runs']:
                    advanced_args['model.pretrained_exp_root'] = gather_exp_paths(str(inf_group_dir)), # Note this is a list of train exp paths.
                else:
                    advanced_args['model.pretrained_exp_root'] = [str(inf_group_dir)]
                # Combine the default and advanced arguments.
                default_config_options.update(advanced_args)
                # Append these to the list of configs and roots.
                calibrator_option_list.append(default_config_options)

    # Return the list of different configs.
    return calibrator_option_list


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