# ionpy imports
from ionpy import slite
from ionpy.util import Config
from ionpy.util.config import config_digest
from ionpy.util.ioutil import autosave
from ionpy.experiment.util import generate_tuid

# local imports
from .ese_exp import CalibrationExperiment 

# misc imports
import os
import pathlib
from typing import List, Optional, Union
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_ese_exp(
    config: Config,
    gpu: str,
    run_name: str = None,
    show_examples: bool = False,
    wandb: bool = False
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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_ese_exps(
    exp_name: str,
    config_list: List[Config],
    available_gpus: List[str],
    wandb: bool = False
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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def restart_ese_exps(
    exp_name: str,
    job_names: Union[str, List[str]],
    available_gpus: List[str],
    modifications: Optional[dict] = None,
    wandb: bool = False,
    exp_root: pathlib.Path = pathlib.Path("/storage/vbutoi/scratch/ESE")
):
    exp_dir = exp_root / exp_name
    # Either rerun the entire experiment, or specific jobs.
    if job_names == "all":
        exp_path_files = [jn for jn in list(exp_dir.iterdir()) if jn.name not in ["submitit", "wandb"]]
    else:
        exp_path_files = [exp_dir / job_name for job_name in job_names]
    
    # Load the configs.
    config_list = [Config.load(exp_path / "config.yml") for exp_path in exp_path_files]
        
    # Get the config as a dictionary.
    for exp_path, config in zip(exp_path_files, config_list):
        cfg = config.to_dict()

        # Remove the step callback because it will slow down training.
        if "step" in cfg["callbacks"]:
            cfg["callbacks"].pop("step")

        # If you want wandb logging, and it is not already in the config, add it.
        if wandb and ("ese.experiment.callbacks.WandbLogger" not in cfg["callbacks"]["epoch"]):
            cfg["callbacks"]["epoch"].append("ese.experiment.callbacks.WandbLogger")

        # If you don't want wandb logging, and it is in the config, remove it.
        if not wandb and ("ese.experiment.callbacks.WandbLogger" in cfg["callbacks"]["epoch"]):
            cfg["callbacks"]["epoch"].remove("ese.experiment.callbacks.WandbLogger")
        
        # Update the config with any modifications
        if modifications:
            cfg = Config(cfg).update(modifications).to_dict()

        # Create some new metadata
        create_time, nonce = generate_tuid()
        digest = config_digest(cfg)
        metadata = {"create_time": create_time, "nonce": nonce, "digest": digest}

        # Move the old config and metadata to backup files if they don't exist.
        if not (exp_path / "metadata.json.backup").is_dir():
            os.rename(exp_path / "metadata.json", exp_path / "metadata.json.backup")
        if not (exp_path / "config.yml.backup").is_dir():
            os.rename(exp_path / "config.yml", exp_path / "config.yml.backup")

        # Save the new config and metadata
        autosave(metadata, exp_path / "metadata.json")
        autosave(cfg, exp_path / "config.yml")
        
    # Run the experiments 
    slite.submit_exps(
        project="ESE",
        exp_name=exp_name,
        exp_class=CalibrationExperiment,
        exp_path_list=exp_path_files,
        available_gpus=available_gpus
    ) 