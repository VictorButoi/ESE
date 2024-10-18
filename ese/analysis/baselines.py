# Misc imports
from typing import Optional
# Local imports
from ese.analysis.analyze_inf import load_cal_inference_stats


def add_baseline_lines(
    g_plot,
    data_df,
    baseline_dict,
    y,
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
            "inference_groups": []
        },
        "options": {
            "verify_graceful_exit": True,
            "equal_rows_per_cfg_assert": False 
        }
    }

    # We can add to the plot the naive baselines which are
    # - just the default sum up probs 
    # - threshold at 0.5.
    if add_naive_baselines:
        naive_result_cfg = results_cfgs.copy()
        naive_result_cfg['log']['inference_groups'] = [
            "Base_CrossEntropy",
            "Base_SoftDice"
        ]
        # Load the naive baselines
        naive_inference_df = load_cal_inference_stats(
            results_cfg=naive_result_cfg,
            load_cached=True
        )

    # Then we have another set of baselines that are when we tune the
    # on the calibratio set for the 
    # - hard threshold
    # - soft temperature
    if add_tuned_baselines:
        tuned_result_cfg = results_cfgs.copy()
        tuned_result_cfg['log']['inference_groups'] = [
            "Tuned_CrossEntropy",
            "Tuned_SoftDice"
        ]
        # Load the tuned baselines
        tuned_inference_df = load_cal_inference_stats(
            results_cfg=tuned_result_cfg,
            load_cached=True
        )
    

