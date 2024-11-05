# Misc imports
from pathlib import Path
from pprint import pprint
from pydantic import validate_arguments
# Local imports
from .parse_sweep import get_global_optimal_parameter
from ..analyze_inf import load_cal_inference_stats


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def add_sweep_options(
    total_cfg: dict, 
    param: str,
):
    # If we are doing a  sweep then we just have a single parameter instead of a val func.
    if param.lower() == "threshold":
        exp_cfg_update = {
            "inf_kwargs": {
                "threshold": [ 
                    "(0.00, ..., 0.25, 0.025)",
                    "(0.26, ..., 0.50, 0.025)",
                    "(0.51, ..., 0.75, 0.025)",
                    "(0.76, ..., 1.00, 0.025)"
                ]
            }
        }
    elif param.lower() == "temperature":
        exp_cfg_update = {
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
    else:
        raise ValueError(f"Unknown parameter: {param}")

    # Update the experiment config.
    total_cfg['experiment'].update(exp_cfg_update)
    return total_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_sweep_optimal_params(
    log_root: Path,
    id_key: str,
    split: str,
    metric: str,
    sweep_key: str,
    **opt_kwargs
):
    # Get the optimal parameter for some value function.
    assert sweep_key.lower() in ["threshold", "temperature"], f"Unknown sweep key: {sweep_key}."

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

    # Importantly, if 'split' is not defined then we need to set it as the default
    if 'split' not in sweep_df.columns:
        sweep_df['split'] = sweep_df['inference_data_split']

    # Do to artifact of the post-processing, we need to parse the sweep key to match how we saved them.
    parsed_id_key = id_key.replace(".", "_")
    # Get the optimal parameter for the inference.
    all_param_opt_vals = get_global_optimal_parameter(
        sweep_df,
        sweep_key=sweep_key, 
        y_key=metric,
        group_keys=[parsed_id_key, 'split'],
        **opt_kwargs
    )
    # Get only the rows where split is cal.
    param_opt_vals = all_param_opt_vals[all_param_opt_vals['split'] == split]
    # We now need to ranme the parsed_id_key column to go back to the original id_key.
    param_opt_vals = param_opt_vals.rename(columns={parsed_id_key: id_key})
    # We want to make a dictionary that maps from 'experiment_model_dir' to the optimal parameter.
    opt_param_dict = dict(zip(param_opt_vals[id_key], param_opt_vals[sweep_key]))
    # Make a dictionary that maps from the model_dir to the proper experiment config update.
    return {model_dir: {f"experiment.inf_kwargs.{sweep_key}": opt_param} for model_dir, opt_param in opt_param_dict.items()}