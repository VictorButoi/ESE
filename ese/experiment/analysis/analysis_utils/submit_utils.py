# misc imports
import itertools
from pathlib import Path
from typing import List, Optional
# ESE imports
from ese.scripts.utils import gather_exp_paths
from ese.experiment.models.utils import get_calibrator_cls


def get_ese_inference_configs(
    group_dict: dict,
    calibrators_list: List[str], 
    ensemble_opts: List[bool],
    log_image_stats: bool = True,
    log_pixel_stats: bool = True,
    ensemble_upper_bound: bool = False,
    num_ens_members_opts: Optional[List[int]] = [None],
    norm_ens_opts: Optional[List[bool]] = [False],
    norm_binning_opts: Optional[List[bool]] = [False],
    cal_stats_splits: Optional[List[str]] = [None],
    additional_args: Optional[dict] = None,
):
    scratch_root = Path("/storage/vbutoi/scratch/ESE")
    # Gather the different config options.
    run_cfg_options = list(itertools.product(
        calibrators_list, 
        ensemble_opts, 
        norm_ens_opts,
        norm_binning_opts,
        cal_stats_splits,
    ))
    # Keep a list of all the run configuration options.
    calibrator_option_list = []
    # Using itertools, get the different combos of calibrators_list ens_cfg_options and ens_w_metric_list.
    for (calibrator, do_ensemble, norm_ensemble, norm_binning, cal_stats_split) in run_cfg_options: 
        ens_cfg_options = [('mean', 'probs'), ('product', 'probs')] if do_ensemble else [None]
        # For each ensemble option, we want to run inference.
        for ens_cfg in ens_cfg_options:
            ##################################################
            # Set a few things that will be consistent for all runs.
            ##################################################
            exp_root = scratch_root / "inference" / group_dict['exp_group']
            use_uncalibrated_models = (calibrator == "Uncalibrated") or ("Binning" in calibrator)
            # Define the set of default config options.
            default_config_options = {
                'experiment.exp_root': [str(exp_root)],
                'experiment.dataset_name': [group_dict['dataset']],
                'model.checkpoint': ["max-val-dice_score" if use_uncalibrated_models else "min-val-ece_loss"],
                'model.calibrator': [calibrator],
                'model.calibrator_cls': [get_calibrator_cls(calibrator)],
                'log.log_image_stats': [log_image_stats],
                'log.log_pixel_stats': [log_pixel_stats],
            }
            if 'preload' in group_dict:
                default_config_options['data.preload'] = [group_dict['preload']]

            # Add the unique arguments for the binning calibrator.
            if "Binning" in calibrator:
                default_config_options['model.normalize'] = [norm_binning]
                default_config_options['model.cal_stats_split'] = [cal_stats_split]
            # If additional args are provided, update the default config options.
            if additional_args is not None:
                default_config_options.update(additional_args)

            # Define where we get the base models from.
            if use_uncalibrated_models:
                inf_group_dir = scratch_root / "training" / group_dict['base_models_group']
            else:
                inf_group_dir = scratch_root / "calibration" / group_dict['calibrated_models_group'] / f"Individual_{calibrator}"

            # If you want to run inference on ensembles, use this.
            if do_ensemble:
                # Define where we want to save the results.
                if ensemble_upper_bound:
                    inf_log_root = exp_root / f"ensemble_upper_bounds"
                else:
                    inf_log_root = exp_root / f"{group_dict['dataset']}_Ensemble_{calibrator}"
                # Subselect the model names in inf_grou_dir
                total_ens_members = gather_exp_paths(str(inf_group_dir))
                # For each num_ens_members, we subselect that num of the total_ens_members.
                for num_ens_members in num_ens_members_opts:
                    # Get all unique subsets of total_ens_members of size num_+ens_members.
                    unique_ensembles = list(itertools.combinations(total_ens_members, num_ens_members))
                    for ens_group in unique_ensembles:
                        # Define where the set of base models come from.
                        advanced_args = {
                            'log.root': [str(inf_log_root)],
                            'model.ensemble': [True],
                            'ensemble.normalize': [norm_ensemble],
                            'ensemble.combine_fn': [ens_cfg[0]],
                            'ensemble.combine_quantity': [ens_cfg[1]],
                            'ensemble.member_paths': [list(ens_group)],
                        }
                        # Combine the default and advanced arguments.
                        default_config_options.update(advanced_args)
                        # Append these to the list of configs and roots.
                        calibrator_option_list.append(default_config_options)
            # If you want to run inference on individual networks, use this.
            else:
                advanced_args = {
                    'log.root': [str(exp_root / f"{group_dict['dataset']}_Individual_{calibrator}")],
                    'model.pretrained_exp_root': gather_exp_paths(str(inf_group_dir)), # Note this is a list of train exp paths.
                    'model.ensemble': [False],
                    'ensemble.normalize': [None],
                    'ensemble.combine_fn': [None],
                    'ensemble.combine_quantity': [None],
                }
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