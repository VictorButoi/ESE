import pickle
from ionpy.util import dict_product
from ionpy.slite.utils import proc_exp_name, validate_cfg

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

def get_option_product(exp_name, option_set, base_cfg, ):
    cfgs = []
    for option_dict in option_set:
        for cfg_update in dict_product(option_dict):
            cfg = base_cfg.update(cfg_update)
            cfg = cfg.update(proc_exp_name(exp_name, cfg_update))
            cfg = validate_cfg(cfg)
            cfgs.append(cfg)
    return cfgs