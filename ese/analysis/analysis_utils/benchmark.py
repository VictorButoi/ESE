# Misc imports
from pathlib import Path
from pprint import pprint
from pydantic import validate_arguments
# Local imports
from .parse_sweep import get_global_optimal_parameter
from ..analyze_inf import load_cal_inference_stats


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def add_sweep_options(
    experiment_cfg: dict, 
    determiner: str,
):
    _, parameters = determiner.split("_")[0], determiner.split("_")[1]
    # If we are doing a  sweep then we just have a single parameter instead of a val func.
    param = parameters[0].lower()
    if param == "threshold":
        exp_cfg_update = {
            "experiment": {
                "inf_kwargs": {
                    "threshold": [ 
                        "(0.00, ..., 0.25, 0.025)",
                        "(0.26, ..., 0.50, 0.025)",
                        "(0.51, ..., 0.75, 0.025)",
                        "(0.76, ..., 1.00, 0.025)"
                    ]
                }
            }
        }
    elif param == "temperature":
        exp_cfg_update = {
            "experiment": {
                "inf_kwargs": {
                    "temperature": [ 
                        "(0.01, ..., 0.50, 0.025)",
                        "(0.51, ..., 1.00, 0.025)", 
                        "(1.01, ..., 1.25, 0.025)",
                        "(1.26, ..., 1.50, 0.025)",
                        "(1.51, ..., 2.00, 0.025)",
                        "(2.01, ..., 2.50, 0.025)",
                        "(2.51, ..., 2.75, 0.025)",
                        "(2.76, ..., 3.00, 0.025)",
                    ]
                }
            }
        }
    else:
        raise ValueError(f"Unknown parameter: {param}")

    # Update the experiment config.
    experiment_cfg.update(exp_cfg_update)

    return experiment_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_sweep_optimal_params(
    determiner: str,
    log_root: Path
):
    # Get the optimal parameter for some value function.
    _, parameters = determiner.split("_")[0], determiner.split("_")[1:]
    val_func, sweep_key = parameters
    assert sweep_key.lower() in ["threshold", "temperature"], f"Unknown sweep key: {sweep_key}."

    parsed_sweep_key = sweep_key.lower()
    parsed_y_key = f"soft_{val_func}" if parsed_sweep_key == "temperature" else f"hard_{val_func}"
    # Load the sweep directory.
    results_cfgs = {
        "log":{
            "root": str(log_root.parent),
            "inference_group": f"Sweep_{sweep_key}"
        },
        "options": {
            "verify_graceful_exit": True,
            "equal_rows_per_cfg_assert": False 
        }
    }

    # Load the naive baselines
    print(f"Loading {sweep_key} sweep dataframe...")
    sweep_df = load_cal_inference_stats(
        results_cfg=results_cfgs,
        load_cached=True
    )

    # Get the optimal parameter for the inference.
    all_param_opt_vals = get_global_optimal_parameter(
        sweep_df,
        sweep_key=parsed_sweep_key, 
        y_key=parsed_y_key,
        group_keys=['experiment_model_dir', 'split'] 
    )

    # Get only the rows where split is cal.
    param_opt_vals = all_param_opt_vals[all_param_opt_vals['split'] == 'cal']

    # We want to make a dictionary that maps from 'experiment_model_dir' to the optimal parameter.
    cal_opt_param_dict = dict(zip(param_opt_vals['experiment_model_dir'], param_opt_vals[parsed_sweep_key]))

    # Make a dictionary that maps from the model_dir to the proper experiment config update.
    return {model_dir: {f"experiment.inf_kwargs.{parsed_sweep_key}": cal_opt_param} for model_dir, cal_opt_param in cal_opt_param_dict.items()}