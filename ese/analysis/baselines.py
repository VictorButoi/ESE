# Misc imports
import pandas as pd
from typing import Optional
# Local imports
from .analyze_inf import load_cal_inference_stats
from .analysis_utils.parse_sweep import get_global_optimal_parameter 


def gather_baseline_dfs(
    data_df,
    baseline_dict,
    add_naive_baselines: bool = True,
    add_tuned_baselines: bool = True
):
    # First, we need to get the dataset that the df corresponds to
    unique_datasets = data_df['data'].unique()
    # Assert that we only have one dataset
    assert len(unique_datasets) == 1
    dataset = unique_datasets[0].split('.')[-1]

    # Next, we use the baseline_dict to get the baselines
    root_baseline_dir = baseline_dict[dataset]
    results_cfgs = {
        "log":{
            "root": root_baseline_dir,
            "inference_group": []
        },
        "options": {
            "verify_graceful_exit": True,
            "equal_rows_per_cfg_assert": False 
        }
    }

    # Make a dict to store the baselines
    baselines = {}

    # We can add to the plot the naive baselines which are
    # - just the default sum up probs 
    # - threshold at 0.5.
    if add_naive_baselines:
        naive_result_cfg = results_cfgs.copy()
        naive_result_cfg['log']['inference_group'] = [
            "Base_CrossEntropy",
            "Base_SoftDice"
        ]
        # Load the naive baselines
        naive_inference_df = load_cal_inference_stats(
            results_cfg=naive_result_cfg,
            load_cached=True
        )
        baselines.update({"base": naive_inference_df})

    # Then we have another set of baselines that are when we tune the
    # on the calibratio set for the 
    # - hard threshold
    # - soft temperature
    if add_tuned_baselines:
        # Load the threshold baselines.
        tuned_threshold_cfg = results_cfgs.copy()
        tuned_threshold_cfg['log']['inference_group'] = [
            "Optimal_Temperature_CrossEntropy",
            "Optimal_Temperature_SoftDice",
            "Optimal_Threshold_CrossEntropy",
            "Optimal_Threshold_SoftDice"
        ]
        # Load the tuned baselines
        tuned_threshold_df = load_cal_inference_stats(
            results_cfg=tuned_threshold_cfg,
            load_cached=True
        )

        # Load the temperatures baselines.
        tuned_temp_cfg = results_cfgs.copy()
        tuned_temp_cfg['log']['inference_group'] = [
            "Optimal_Temperature_CrossEntropy",
            "Optimal_Temperature_SoftDice",
            "Optimal_Threshold_CrossEntropy",
            "Optimal_Threshold_SoftDice"
        ]
        # Load the tuned baselines
        tuned_temperature_df = load_cal_inference_stats(
            results_cfg=tuned_temp_cfg,
            load_cached=True
        )
        baselines.update(
            {
                "threshold": tuned_threshold_df,
                "temperature": tuned_temperature_df
            }
        )
    
    return baselines


def get_baseline_values(y_key, baselines_dict):

    method_dfs = []

    if 'threshold' in baselines_dict:
        thresh_opt_vals = get_global_optimal_parameter(
            baselines_dict['threshold'], 
            sweep_key='threshold', 
            y_key="hard_"+y_key,
            group_keys=['split', 'loss_func_class']
        )
        # Rename the y_key to be the same as the input y_key
        thresh_opt_vals.rename(columns={f"hard_{y_key}": y_key}, inplace=True)
        thresh_opt_vals['method'] = "threshold_tuning"
        method_dfs.append(thresh_opt_vals)

    if 'temperature' in baselines_dict:
        temp_opt_vals = get_global_optimal_parameter(
            baselines_dict['temperature'], 
            sweep_key='temperature', 
            y_key="soft_"+y_key,
            group_keys=['split', 'loss_func_class']
        )
        # Rename the y_key to be the same as the input y_key
        temp_opt_vals.rename(columns={f"soft_{y_key}": y_key}, inplace=True)
        # Move it to be the last column
        temp_opt_vals['method'] = "temperature_tuning"
        method_dfs.append(temp_opt_vals)
    
    # Concatenate the dataframes and return
    return pd.concat(method_dfs)
    

