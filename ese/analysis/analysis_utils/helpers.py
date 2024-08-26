# misc imports
import os
import yaml
from pprint import pprint
from datetime import datetime
from itertools import chain, combinations
# Ionpy imports
from ionpy.util import Config
from ionpy.util import dict_product
from ionpy.util.ioutil import autosave
from ionpy.util.config import check_missing, HDict, valmap


def gather_exp_paths(root):
    # For ensembles, define the root dir.
    run_names = os.listdir(root)
    # NOTE: Not the best way to do this, but we need to skip over some files/directories.
    skip_items = [
        "submitit",
        "wandb",
        "base.yml",
        "experiment.yml"
    ]
    # Filter out the skip_items
    valid_exp_paths = []
    for run_name in run_names:
        run_dir = f"{root}/{run_name}"
        # Make sure we don't include the skip items and that we actually have valid checkpoints.
        if (run_name not in skip_items) and os.path.isdir(f"{run_dir}/checkpoints"):
            valid_exp_paths.append(run_dir)
    # Return the valid experiment paths.
    return valid_exp_paths


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


def get_option_product(
    exp_name,
    option_set,
    base_cfg
):
    # If option_set is not a list, make it a list
    cfgs = []
    # Get all of the keys that have length > 1 (will be turned into different options)
    varying_keys = [key for key, value in option_set.items() if len(value) > 1]
    # Iterate through all of the different options
    for cfg_update in dict_product(option_set):
        # If one of the keys in the update is a dictionary, then we need to wrap
        # it in a list, otherwise the update will collapse the dictionary.
        for key in cfg_update:
            if isinstance(cfg_update[key], dict):
                cfg_update[key] = [cfg_update[key]]
        # Get the name that will be used for WANDB tracking and update the base with
        # this version of the experiment.
        cfg_name_args = proc_cfg_name(exp_name, varying_keys, cfg_update)
        cfg = base_cfg.update([cfg_update, cfg_name_args])
        # Verify it's a valid config
        check_missing(cfg)
        cfgs.append(cfg)
    return cfgs


def listify_dict(d):
    listy_d = {}
    # We need all of our options to be in lists as convention for the product.
    for ico_key in d:
        # If this is a tuple, then convert it to a list.
        if isinstance(d[ico_key], tuple):
            listy_d[ico_key] = list(d[ico_key])
        # Otherwise, make sure it is a list.
        elif not isinstance(d[ico_key], list):
            listy_d[ico_key] = [d[ico_key]]
        else:
            listy_d[ico_key] = d[ico_key]
    # Return the listified dictionary.
    return listy_d


def flatten_cfg2dict(cfg: Config):
    cfg = HDict(cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    return flat_exp_cfg


def power_set(in_set):
    return list(chain.from_iterable(combinations(in_set, r) for r in range(len(in_set)+1)))


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def save_exp_cfg(exp_cfg, exp_name, group, add_date, scratch_root):
    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        exp_name = f"{formatted_date}_{exp_name}"
    # Save the experiment config.
    exp_root = scratch_root / group / exp_name
    autosave(exp_cfg, exp_root / "experiment.yml") # SAVE #1: Experiment config
    # Return the experiment root.
    return exp_root

