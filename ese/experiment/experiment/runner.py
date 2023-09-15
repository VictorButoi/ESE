from ionpy import slite
from ionpy.util import Config
from ese.experiment.experiment.ese_exp import CalibrationExperiment 


def run_ese_exp(
    config,
    gpu,
    show_examples=False,
    wandb=False
):
    # Get the config as a dictionary.
    cfg = config.to_dict()

    if not show_examples:
        cfg["callbacks"]["step"].pop("ese.experiment.callbacks.ShowPredictions")

    if not wandb:
        cfg["callbacks"]["epoch"].pop("ese.experiment.callbacks.WandbCallback")

    # Run the experiment.
    slite.run_exp(
        config=Config(cfg), 
        exp_class=CalibrationExperiment,
        gpu=gpu,
    )
    