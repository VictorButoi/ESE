from ionpy import slite
from ionpy.util import Config
from ese.experiment.experiment.ese_exp import CalibrationExperiment 


def run_ese_exp(
    config,
    gpu,
    run_name=None,
    show_examples=False,
    wandb=False
):
    # Get the config as a dictionary.
    cfg = config.to_dict()

    if run_name is not None:
        log_dir_root = "/".join(cfg["log"]["root"].split("/")[:-1])
        cfg["log"]["root"] = "/".join([log_dir_root, run_name])

    if not show_examples:
        cfg["callbacks"].pop("step")

    if not wandb:
        cfg["callbacks"]["epoch"].remove("ese.experiment.callbacks.WandbLogger")

    # Run the experiment.
    slite.run_exp(
        config=Config(cfg), 
        exp_class=CalibrationExperiment,
        gpu=gpu,
    )

def submit_ese_exps(
    exp_name,
    config_list,
    available_gpus,
    wandb=False
):
    # Get the config as a dictionary.
    for config in config_list:
        cfg = config.to_dict()
        # Remove the step callback because it will slow down training.
        cfg["callbacks"].pop("step")
        if not wandb:
            cfg["callbacks"]["epoch"].remove("ese.experiment.callbacks.WandbLogger")

    # Run the experiments 
    slite.submit_exps(
        project="ESE",
        exp_name=exp_name,
        exp_class=CalibrationExperiment,
        available_gpus=available_gpus,
        config_list=config_list
    ) 