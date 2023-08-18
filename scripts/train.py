# Submit cell
from ese.experiment.experiment.ese_exp import CalibrationExperiment 
from ionpy.slite import SliteRunner
from ionpy.util.config import check_missing
from ionpy.util import dict_product, Config

# Random imports
import pickle
import os

def validate_cfg(cfg):
    # It's usually a good idea to do a sanity check of
    # inter-related settings or force them manually
    check_missing(cfg)        
    return cfg

if __name__=="__main__":
    # Regular schema dictates that we put DATAPATH
    os.environ['DATAPATH'] = ':'.join((
        '/storage/vbutoi/datasets',
    ))
    log_root_dir = '/storage/vbutoi/scratch/ESE'

    # Assemble base config
    base_cfg = None # Load the default config we build in notebook
    with open('/storage/vbutoi/projects/ESE/configs/base.pkl', 'rb') as config_file:
        base_cfg = pickle.load(config_file)

    # Need to define the experiment name
    exp_name = 'ablation_slice_batch_size'

    # Create the ablation options
    option_set = [
        {
            'log.root': [f'{log_root_dir}/{exp_name}'],
            'dataloader.num_workers': [4],
            'data.slice_batch_size': [16, 24, 32],
            'model.filters': [[32, 32, 32, 32], [64, 64, 64, 64]]
        }
    ]

    cfgs = []
    for option_dict in option_set:
        for cfg_update in dict_product(option_dict):
            cfg = base_cfg.update(cfg_update)
            cfg = validate_cfg(cfg)
            cfgs.append(cfg)

    # List the available gpus for a machine
    available_gpus = ['0', '1', '2', '3']

    # Configure Slite Object
    srunner = SliteRunner(
        task_type=CalibrationExperiment, 
        exp_name=exp_name, 
        available_gpus=available_gpus
        )

    srunner.submit_exps(cfgs)