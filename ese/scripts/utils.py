# random imports
import os
import pickle
# ionpy imports
from ionpy.util.config import check_missing
from ionpy.util import dict_product


def LOAD_BASE_CONFIG():
    # Load the default config we build in notebook
    base_cfg = None
    with open('/storage/vbutoi/projects/ESE/configs/base.pkl', 'rb') as config_file:
        base_cfg = pickle.load(config_file)
    return base_cfg


def LOAD_AUG_CONFIG():
    # Load the default config we build in notebook
    lite_aug_cfg = None
    with open('/storage/vbutoi/projects/ESE/configs/lite_aug.pkl', 'rb') as config_file:
        lite_aug_cfg = pickle.load(config_file)
    return lite_aug_cfg


def gather_exp_paths(root):
    # For ensembles, define the root dir.
    run_names = os.listdir(root)
    if "submitit" in run_names:
        run_names.remove("submitit")
    if "wandb" in run_names:
        run_names.remove("wandb")
    run_names = [f"{root}/{run_name}" for run_name in run_names]
    return run_names


def get_option_product(
    exp_name,
    option_set,
    base_cfg
):
    # If option_set is not a list, make it a list
    if not isinstance(option_set, list):
        option_set = [option_set]
    cfgs = []
    for option_dict in option_set:
        # Get all of the keys that have length > 1 (will be turned into different options)
        varying_keys = [key for key, value in option_dict.items() if len(value) > 1]
        # Iterate through all of the different options
        for cfg_update in dict_product(option_dict):
            cfg_name_args = proc_cfg_name(exp_name, varying_keys, cfg_update)
            cfg = base_cfg.update([cfg_update, cfg_name_args])
            # Verify it's a valid config
            check_missing(cfg)
            cfgs.append(cfg)
    return cfgs


def proc_cfg_name(
    exp_name,
    varying_keys,
    cfg
):
    params = []
    params.append("exp_name:" + exp_name)
    for key, value in cfg.items():
        if key in varying_keys:
            if key not in ["log.root", "train.pretrained_dir"]:
                key_name = key.split(".")[-1]
                short_value = str(value).replace(" ", "")
                if key_name == "exp_name":
                    params.append(str(short_value))
                else:
                    params.append(f"{key_name}:{short_value}")
    wandb_string = "-".join(params)
    return {"log.wandb_string": wandb_string}
