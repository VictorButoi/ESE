# Submit cell
from ese.experiment.experiment.ese_exp import CalibrationExperiment 
from .utils import LOAD_BASE_CONFIG, LOAD_AUG_CONFIG, get_option_product 
from ionpy.slite import SliteRunner

# Random imports
import pickle
import copy
import os


if __name__=="__main__":
    # Regular schema dictates that we put DATAPATH
    os.environ['DATAPATH'] = ':'.join((
        '/storage/vbutoi/datasets',
    ))

    # Configure Slite Object
    srunner = SliteRunner(
        project='ESE',
        task_type=CalibrationExperiment, 
        available_gpus=['0', '1', '2', '3']
        )

    # Assemble base config
    base_cfg = None # Load the default config we build in notebook
    with open('/storage/vbutoi/projects/ESE/configs/base.pkl', 'rb') as config_file:
        base_cfg = pickle.load(config_file)

    # Need to define the experiment name
    exp_name = 'FixedSizeAblation'

    # Create the ablation options
    option_set = [
        {
            'log.root': [f'/storage/vbutoi/scratch/ESE/{exp_name}'],
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

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg)
    
    # Add Augmentations
    for cfg_idx in range(len(cfgs)):
        for add_aug in [True, False]:
            if add_aug:
                cfgs[cfg_idx] = cfgs[cfg_idx].set('augmentations', sum([copy.deepcopy(lite_aug_cfg)], start=[]))

    # Submit the experiments
    srunner.set_exp_name(exp_name)
    srunner.submit_exps(cfgs)