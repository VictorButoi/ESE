# Misc imports
from pathlib import Path
from pprint import pprint
from pydantic import validate_arguments
# Local imports
from .parse_sweep import get_global_optimal_parameter
from ..analyze_inf import load_cal_inference_stats


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_benchmark_params(
    experiment_cfg: dict, 
    determiner: str,
    log_root: Path
):
        inference_type, parameters = determiner.split("_")[0], determiner.split("_")[1:]
        # These allow us sweep over the parameters for the inference.
        if inference_type.lower() == "sweep":
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
                raise ValueError(f"Unknown parameter for sweep inference: {param}.")
        # We run this after we have the sweeps complete to automatically parse the optimal parameters.
        elif inference_type.lower() == "optimal":
            # Get the optimal parameter for some value function.
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
            # We are going to use base_model in the exp_cfg to select the rows we want.
            base_model = experiment_cfg["base_model"]
            assert isinstance(base_model, str) or len(base_model) == 1, "Base model must be a single model."
            if isinstance(base_model, list):
                base_model = base_model[0]
            # Get the rows correspondign to the base model.
            base_model_sweep_df = sweep_df[sweep_df['experiment_model_dir'] == base_model].copy()
            # Get the optimal parameter for the inference.
            param_opt_vals = get_global_optimal_parameter(
                base_model_sweep_df,
                sweep_key=parsed_sweep_key, 
                y_key=parsed_y_key,
                group_keys=['split'] 
            )
            # Get the optimal parameter for the inference for the 
            # calibration split
            cal_opt_param = float(param_opt_vals[param_opt_vals['split'] == 'cal'][parsed_sweep_key].values[0])
            # Update with the optimal parameter for the inference.
            exp_cfg_update = {
                "experiment": {
                    "inf_kwargs": {
                        parsed_sweep_key: cal_opt_param
                    }
                }
            }
        else:
            exp_cfg_update = {}
        
        # Update the experiment config.
        experiment_cfg.update(exp_cfg_update)

        return experiment_cfg