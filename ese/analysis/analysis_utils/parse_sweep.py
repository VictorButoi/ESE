# misc imports
import pandas as pd
from typing import List, Optional
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_global_optimal_parameter(
    data: pd.DataFrame, 
    sweep_key: str, 
    y_key: str,
    group_keys: Optional[List[str]] = [] 
) -> pd.DataFrame:
    # Get the optimal threshold for each split out. First we have to average across the data_ids
    reduced_data_df = data.groupby(group_keys + [sweep_key]).mean().reset_index()
    # Then we get the threshold that minimizes the error
    optimal_df = reduced_data_df.loc[reduced_data_df.groupby(group_keys)[y_key].idxmin()]
    # Finally, we only keep the columns we care about.
    best_parameter_df = optimal_df[group_keys + [sweep_key, y_key]].reset_index(drop=True).sort_values(y_key)
    # Return the best thresholds
    return best_parameter_df 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_per_subject_optimal_values(
    data: pd.DataFrame, 
    sweep_key: str, 
    y_key: str,
    group_keys: Optional[List[str]] = None
) -> pd.DataFrame:
    # We want to figure out what is the best achievable average loss IF we used optimal thresholds per subject
    sub_cols_to_keep = [
        sweep_key,
        y_key,
        'data_id',
    ]
    if group_keys is not None:
        sub_cols_to_keep += group_keys
    # Filter out the columns we want to keep
    reduced_data_df = data[sub_cols_to_keep].drop_duplicates().reset_index(drop=True)
    # Get the optimal temperature for each data_id
    optimal_df = reduced_data_df.loc[reduced_data_df.groupby('data_id')[y_key].idxmin()].reset_index(drop=True)
    # We want, per split, to get the average loss if we used the optimal temperature for each subject
    if group_keys is not None:
        optimal_df = optimal_df.groupby(group_keys)
    # Mean across all the groups
    opt_values_df = optimal_df.agg({y_key: 'mean'}).reset_index()
    # Return the best values
    return opt_values_df 


