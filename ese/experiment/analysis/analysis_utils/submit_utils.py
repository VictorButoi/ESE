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
    do_ensemble: bool = False, 
    log_image_stats: bool = True,
    log_pixel_stats: bool = True,
    ensemble_upper_bound: bool = False,
    normalize_opts: Optional[List[bool]] = [None],
    discrete_neighbors_opts: Optional[List[bool]] = [None],
    cal_stats_splits: Optional[List[str]] = [None],
    additional_args: Optional[dict] = None,
):
    scratch_root = Path("/storage/vbutoi/scratch/ESE")
    
    # For ensembles, we have three choices for combining the predictions.
    if do_ensemble:
        ens_cfg_options=[
            ('mean', 'logits'), 
            ('mean', 'probs'), 
            ('product', 'probs')
        ]
    else:
        ens_cfg_options=[None]
    # Keep a list of all the run configuration options.
    calibrator_option_list = []
    # Gather the different config options.
    run_cfg_options = list(itertools.product(
        calibrators_list, 
        ens_cfg_options, 
        normalize_opts, 
        cal_stats_splits,
        discrete_neighbors_opts,
    ))
    # Using itertools, get the different combos of calibrators_list ens_cfg_options and ens_w_metric_list.
    for (calibrator, ens_cfg, normalize, cal_stats_split, discrete_neighbs) in run_cfg_options: 
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
            'data.preload': [group_dict['preload']],
            'log.log_image_stats': [log_image_stats],
            'log.log_pixel_stats': [log_pixel_stats],
        }
        # Add the unique arguments for the binning calibrator.
        if "Binning" in calibrator:
            default_config_options['model.normalize'] = [normalize]
            default_config_options['model.cal_stats_split'] = [cal_stats_split]
            default_config_options['model.discretize_neighbors'] = [discrete_neighbs]
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

            # Define where the set of base models come from.
            advanced_args = {
                'log.root': [str(inf_log_root)],
                'model.pretrained_exp_root': [str(inf_group_dir)],
                'model.ensemble': [True],
                'ensemble.combine_fn': [ens_cfg[0]],
                'ensemble.combine_quantity': [ens_cfg[1]],
            }
        # If you want to run inference on individual networks, use this.
        else:
            advanced_args = {
                'log.root': [str(exp_root / f"{group_dict['dataset']}_Individual_{calibrator}")],
                'model.pretrained_exp_root': gather_exp_paths(str(inf_group_dir)), # Note this is a list of train exp paths.
                'model.ensemble': [False],
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
    base_options: Optional[dict] = None
):
    scratch_root = Path("/storage/vbutoi/scratch/ESE")
    training_exps_dir = scratch_root / "training" / group_dict['base_models_group']

    cal_option_list = []
    for calibrator in calibrators:
        log_root = scratch_root / 'calibration' / group_dict['exp_group'] / f"Individual_{calibrator}"
        # Get the calibrator name
        calibrator_class_name_map = {
            "LTS": "ese.experiment.models.calibrators.LTS",
            "TempScaling": "ese.experiment.models.calibrators.Temperature_Scaling",
            "VectorScaling": "ese.experiment.models.calibrators.Vector_Scaling",
            "DirichletScaling": "ese.experiment.models.calibrators.Dirichlet_Scaling",
            "NectarScaling": "ese.experiment.models.calibrators.NECTAR_Scaling",
            "NS_V2": "ese.experiment.models.calibrators.NS_V2",
        }
        if calibrator in calibrator_class_name_map:
            calibrator = calibrator_class_name_map[calibrator]

        calibration_options = {
            'log.root': [str(log_root)],
            'data.preload': [group_dict['preload']],
            'train.pretrained_dir': gather_exp_paths(training_exps_dir),
            'model._class': [calibrator],
        }
        if base_options is not None:
            calibration_options.update(base_options)
        # Add the calibration options to the list
        cal_option_list.append(calibration_options)
    # Return the list of calibration options
    return cal_option_list