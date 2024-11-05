# misc imports
import os
import yaml
import itertools
from pathlib import Path
from pprint import pprint
from typing import List, Optional
from pydantic import validate_arguments
# Ionpy imports
from ionpy.util import Config
from ionpy.util import dict_product
from ionpy.util.config import check_missing
# Local imports
from .helpers import *
from .benchmark import add_sweep_options, load_sweep_optimal_params


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ese_training_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    train_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/configs/training"),
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    train_exp_root = get_exp_root(exp_name, group="training", add_date=add_date, scratch_root=scratch_root)

    # Flatten the experiment config.
    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)

    # Add the dataset specific details.
    train_dataset_name = flat_exp_cfg_dict['data._class'].split('.')[-1]
    dataset_cfg_file = train_cfg_root/ f"{train_dataset_name}.yaml"
    with open(dataset_cfg_file, 'r') as d_file:
        dataset_train_cfg = yaml.safe_load(d_file)
    # Update the base config with the dataset specific config.
    base_cfg = base_cfg.update([dataset_train_cfg])
    
    # Get the information about seeds.
    seed = flat_exp_cfg_dict.pop('experiment.seed', 40)
    seed_range = flat_exp_cfg_dict.pop('experiment.seed_range', 1)

    # Create the ablation options.
    option_set = {
        'log.root': [str(train_exp_root)],
        'experiment.seed': [seed + seed_idx for seed_idx in range(seed_range)],
        **listify_dict(flat_exp_cfg_dict)
    }

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg)

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ese_calibration_configs(
    exp_cfg: dict,
    base_cfg: Config,
    calibration_model_cfgs: dict,
    add_date: bool = True,
    code_root: Path = Path("/storage/vbutoi/projects/ESE"),
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE")
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('name')
    calibration_exp_root = get_exp_root(exp_name, group="calibration", add_date=add_date, scratch_root=scratch_root)

    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)
    flat_exp_cfg_dict = listify_dict(flat_exp_cfg_dict) # Make it compatible to our product function.

    cfg_root = code_root / "ese" / "configs" 

    # We need to make sure that these are models and not model folders.
    all_pre_models = []
    for pre_model_dir in flat_exp_cfg_dict['train.base_pretrained_dir']:
        if 'submitit' in os.listdir(pre_model_dir):
            all_pre_models += gather_exp_paths(pre_model_dir) 
        else:
            all_pre_models.append(pre_model_dir)
    # Set it back in the flat_exp_cfg.
    flat_exp_cfg_dict['train.base_pretrained_dir'] = all_pre_models
    
    # Load the dataset specific config and update the base config.
    if 'data._class' in flat_exp_cfg_dict:
        posthoc_dset_name = flat_exp_cfg_dict['data._class'][0].split('.')[-1]
        dataset_cfg_file = cfg_root / "calibrate" / f"{posthoc_dset_name}.yaml"
        # If the dataset specific config exists, update the base config.
        with open(dataset_cfg_file, 'r') as file:
            dataset_cfg = yaml.safe_load(file)
        base_cfg = base_cfg.update([dataset_cfg])
    else:
        _, inf_dset_name = get_inf_dset_from_model_group(flat_exp_cfg_dict['train.base_pretrained_dir'])
        base_cfg = add_dset_presets("calibrate", inf_dset_name, base_cfg, code_root)

    # Create the ablation options.
    option_set = {
        'log.root': [str(calibration_exp_root)],
        **flat_exp_cfg_dict
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
        if isinstance(cal_model, dict):
            model_cfg = calibration_model_cfgs[cal_model.pop('class_name')].copy()
            # Update with the new params and put it back in the cfg.
            model_cfg.update(cal_model)
        else:
            model_cfg = calibration_model_cfgs[cal_model].copy()
        # Put the model cfg back in the cfg_dict.
        cfg_dict['model'] = model_cfg 
        # Replace the Config object with the new config dict.
        cfgs[c_idx] = Config(cfg_dict)

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()

    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ese_inference_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    use_best_models: bool = False,
    power_set_keys: Optional[List[str]] = None,
    code_root: Path = Path("/storage/vbutoi/projects/ESE"),
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE")
):
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    # Save the experiment config.
    group = exp_cfg.pop('group')
    sub_group = exp_cfg.pop('subgroup', "")
    exp_name = f"{group}/{sub_group}"

    # Get the root for the inference experiments.
    inference_log_root = get_exp_root(exp_name, group="inference", add_date=add_date, scratch_root=scratch_root)

    # SPECIAL THINGS THAT GET ADDED BECAUSE WE OFTEN WANT TO DO THE SAME
    # SWEEPS FOR INFERENCE.
    if "sweep" in sub_group.lower(): 
        exp_cfg = add_sweep_options(
            exp_cfg, 
            determiner=sub_group
        )

    # In our general inference sheme, often we want to use the best models corresponding to a dataset
    if add_date:
        eval_dataset = group.split('_')[0]
    else:
        eval_dataset = group.split('_')[3] # Group format is like MM_DD_YY_Dataset
    if eval_dataset in ['OCTA', 'ISLES', "WMH"] and use_best_models:
        # Load the default best models, and update the exp config with those as the base models.
        with open(code_root / "ese" / "configs" / "defaults" / "Best_Models.yaml", 'r') as file:
            best_models_cfg = yaml.safe_load(file)
        # Update the exp_cfg with the best models.
        exp_cfg['base_model'] = best_models_cfg[eval_dataset]
    
    # Flatten the config.
    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)
    # For any key that is a tuple we need to convert it to a list, this is an artifact of the flattening..
    for key, val in flat_exp_cfg_dict.items():
        if isinstance(val, tuple):
            flat_exp_cfg_dict[key] = list(val)

    # Sometimes we want to do a range of values to sweep over, we will know this by ... in it.
    for key, val in flat_exp_cfg_dict.items():
        if isinstance(val, list):
            for idx, val_list_item in enumerate(val):
                if isinstance(val_list_item, str) and '...' in val_list_item:
                    # Replace the string with a range.
                    flat_exp_cfg_dict[key][idx] = get_range_from_str(val_list_item)
        elif isinstance(val, str) and  '...' in val:
            # Finally stick this back in as a string tuple version.
            flat_exp_cfg_dict[key] = get_range_from_str(val)

    # Load the inference cfg from local.
    ##################################################
    default_cfg_root = code_root / "ese" / "configs" / "defaults"
    ##################################################
    with open(default_cfg_root / "Calibration_Metrics.yaml", 'r') as file:
        cal_metrics_cfg = yaml.safe_load(file)
    ##################################################
    base_cfg = base_cfg.update([cal_metrics_cfg])

    # for each power set key, we replace the list of options with its power set.
    if power_set_keys is not None:
        for key in power_set_keys:
            if key in flat_exp_cfg_dict:
                flat_exp_cfg_dict[key] = power_set(flat_exp_cfg_dict[key])

    # Gather the different config options.
    cfg_opt_keys = list(flat_exp_cfg_dict.keys())
    #First going through and making sure each option is a list and then using itertools.product.
    for ico_key in flat_exp_cfg_dict:
        if not isinstance(flat_exp_cfg_dict[ico_key], list):
            flat_exp_cfg_dict[ico_key] = [flat_exp_cfg_dict[ico_key]]
    
    # Generate product tuples 
    product_tuples = list(itertools.product(*[flat_exp_cfg_dict[key] for key in cfg_opt_keys]))

    # Convert product tuples to dictionaries
    total_run_cfg_options = [{cfg_opt_keys[i]: [item[i]] for i in range(len(cfg_opt_keys))} for item in product_tuples]

    # Define the set of default config options.
    default_config_options = {
        'experiment.exp_name': [exp_name],
        'experiment.exp_root': [str(inference_log_root)],
    }
    # Accumulate a set of config options for each dataset
    dataset_cfgs = []
    # Iterate through all of our inference options.
    for run_opt_dict in total_run_cfg_options: 
        # One required key is 'base_model'. We need to know if it is a single model or a group of models.
        # We evaluate this by seeing if 'submitit' is in the base model path.
        base_model_group_dir = Path(run_opt_dict.pop('base_model')[0])
        if 'submitit' in os.listdir(base_model_group_dir):
            model_set  = gather_exp_paths(str(base_model_group_dir)) 
        else:
            model_set = [str(base_model_group_dir)]
        # Append these to the list of configs and roots.
        dataset_cfgs.append({
            'log.root': [str(inference_log_root)],
            'experiment.model_dir': model_set,
            **run_opt_dict,
            **default_config_options
        })

    # SPECIAL THINGS THAT GET ADDED BECAUSE WE OFTEN WANT TO DO THE SAME
    # SWEEPS FOR INFERENCE.
    if "optimal" in sub_group.lower(): 
        optimal_exp_parameters = load_sweep_optimal_params(
            determiner=sub_group,
            log_root=inference_log_root
        )
    else:
        optimal_exp_parameters = None
    
    # Keep a list of all the run configuration options.
    cfgs = []
    # Iterate over the different config options for this dataset. 
    for option_dict in dataset_cfgs:
        for exp_cfg_update in dict_product(option_dict):
            # Add the inference dataset specific details.
            dataset_inf_cfg_dict = get_inference_dset_info(
                cfg=exp_cfg_update,
                code_root=code_root
            )
            # If we have optimal_exp_parameters, then it is per model, so look at the 'experiment.model_dir' key.
            if optimal_exp_parameters is not None:
                exp_cfg_update.update(optimal_exp_parameters[exp_cfg_update['experiment.model_dir']])
            
            # Update the base config with the new options. Note the order is important here, such that 
            # the exp_cfg_update is the last thing to update.
            cfg = base_cfg.update([dataset_inf_cfg_dict, exp_cfg_update])
            # Verify it's a valid config
            check_missing(cfg)
            # Add it to the total list of inference options.
            cfgs.append(cfg)

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ese_restart_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    train_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/configs/training"),
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    restart_exp_root = get_exp_root(exp_name, group="restarted", add_date=add_date, scratch_root=scratch_root)

    # Get the flat version of the experiment config.
    restart_cfg_dict = flatten_cfg2dict(exp_cfg)

    # If we are changing aspects of the dataset, we need to update the base config.
    if 'data._class' in restart_cfg_dict:
        # Add the dataset specific details.
        dataset_cfg_file = train_cfg_root/ f"{restart_cfg_dict['data._class'].split('.')[-1]}.yaml"
        if dataset_cfg_file.exists():
            with open(dataset_cfg_file, 'r') as d_file:
                dataset_train_cfg = yaml.safe_load(d_file)
            # Update the base config with the dataset specific config.
            base_cfg = base_cfg.update([dataset_train_cfg])
        
    # This is a required key. We want to get all of the models and vary everything else.
    pretrained_dir_list = restart_cfg_dict.pop('train.pretrained_dir') 
    if not isinstance(pretrained_dir_list, list):
        pretrained_dir_list = [pretrained_dir_list]

    # Now we need to go through all the pre-trained models and gather THEIR configs.
    all_pre_models = []
    for pre_model_dir in pretrained_dir_list:
        if 'submitit' in os.listdir(pre_model_dir):
            all_pre_models += gather_exp_paths(pre_model_dir) 
        else:
            all_pre_models.append(pre_model_dir)

    # Listify the dict for the product.
    listy_pt_cfg_dict = {
        'log.root': [str(restart_exp_root)],
        **listify_dict(restart_cfg_dict)
    }
    
    # Go through all the pretrained models and add the new options for the restart.
    cfgs = []
    for pt_dir in all_pre_models:
        # Load the pre-trained model config.
        with open(f"{pt_dir}/config.yml", 'r') as file:
            pt_exp_cfg = Config(yaml.safe_load(file))
        # Make a copy of the listy_pt_cfg_dict.
        pt_listy_cfg_dict = listy_pt_cfg_dict.copy()
        pt_listy_cfg_dict['train.pretrained_dir'] = [pt_dir] # Put the pre-trained model back in.
        # Update the pt_exp_cfg with the restart_cfg.
        pt_restart_base_cfg = pt_exp_cfg.update([base_cfg])
        pt_cfgs = get_option_product(exp_name, pt_listy_cfg_dict, pt_restart_base_cfg)
        # Append the list of configs for this pre-trained model.
        cfgs += pt_cfgs

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    return base_cfg_dict, cfgs