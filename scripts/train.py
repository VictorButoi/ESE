# Submit cell
from ese.experiment.experiment.ese_exp import CalibrationExperiment 
from ionpy.slite import SliteRunner
from ionpy.util.config import check_missing
from ionpy.util import dict_product, Config
from ionpy.slite.utils import proc_exp_name

# Random imports
import pickle
import copy
import os

def validate_cfg(cfg):
    # It's usually a good idea to do a sanity check of
    # inter-related settings or force them manually
    check_missing(cfg)        
    return cfg

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

if __name__=="__main__":
    # Regular schema dictates that we put DATAPATH
    os.environ['DATAPATH'] = ':'.join((
        '/storage/vbutoi/datasets',
    ))
    log_root_dir = '/storage/vbutoi/scratch/ESE'

    # List the available gpus for a machine
    available_gpus = ['0', '1', '2', '3']

    # Configure Slite Object
    srunner = SliteRunner(
        project='ESE',
        task_type=CalibrationExperiment, 
        available_gpus=available_gpus
        )

    # Assemble base config
    base_cfg = None # Load the default config we build in notebook
    with open('/storage/vbutoi/projects/ESE/configs/base.pkl', 'rb') as config_file:
        base_cfg = pickle.load(config_file)

    # Need to define the experiment name
    exp_name = 'SizeAblation'

    # Create the ablation options
    option_set = [
        {
            'log.root': [f'{log_root_dir}/{exp_name}'],
            'dataloader.batch_size': [1],
            'data.num_slices' : [4],
            'model.filters': [
                        [128, 128, 128, 128, 128],
                        [256, 256, 256, 256, 256],
                        [512, 512, 512, 512, 512],
                    ],
            'optim.weight_decay': [0, 0.00001, 0.0001],
            'dataloader.num_workers': [2]
        }
    ]

    # Assemble base config
    base_cfg = LOAD_BASE_CONFIG() 
    lite_aug_cfg = LOAD_AUG_CONFIG()

    light_augmentations = sum([copy.deepcopy(lite_aug_cfg)], start=[])

    cfgs = []
    for option_dict in option_set:
        for add_aug in [True, False]:
            for cfg_update in dict_product(option_dict):
                cfg = base_cfg.update(cfg_update)
                cfg = cfg.update(proc_exp_name(exp_name, cfg_update))
                if add_aug:
                    cfg = cfg.set('augmentations', light_augmentations)
                cfg = validate_cfg(cfg)
                cfgs.append(cfg)

    # Submit the experiments
    srunner.set_exp_name(exp_name)
    srunner.submit_exps(cfgs)