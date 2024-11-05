# misc imports
import pandas as pd
from typing import Literal, List, Optional
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_global_optimal_parameter(
    raw_data: pd.DataFrame, 
    sweep_key: str, 
    y_key: str,
    mode: Literal['min', 'max'],
    group_keys: Optional[List[str]] = [] 
) -> pd.DataFrame:
    # Sometimes y_key is not a column, but is one of the possible 'image_metric' values.
    # In such a case, we need to subselect the data to only include the relevant metric.
    if y_key not in raw_data.columns:
        raw_data = raw_data[raw_data['image_metric'] == y_key]
        # Now we can drop the image_metric column, and rename the 'metric_score' column to 'y_key'
        raw_data = raw_data.drop(columns=['image_metric']).rename(columns={'metric_score': y_key})
    # Now we can isolate only the columns we care about.
    cols_to_keep = group_keys + [sweep_key, y_key, 'data_id']
    data = raw_data[cols_to_keep].drop_duplicates().reset_index(drop=True)

    # Get the optimal threshold for each split out. First we have to average across the data_ids
    reduced_data_df = data.groupby(group_keys + [sweep_key]).mean(numeric_only=True).reset_index()
    # Then we get the threshold that minimizes the error
    if mode == 'min':
        optimal_df = reduced_data_df.loc[reduced_data_df.groupby(group_keys)[y_key].idxmin()]
    else:
        optimal_df = reduced_data_df.loc[reduced_data_df.groupby(group_keys)[y_key].idxmax()]
    # Finally, we only keep the columns we care about.
    best_parameter_df = optimal_df[group_keys + [sweep_key, y_key]].reset_index(drop=True).sort_values(y_key)
    # Return the best thresholds
    return best_parameter_df 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_per_subject_optimal_values(
    raw_data: pd.DataFrame, 
    sweep_key: str, 
    y_key: str,
    group_keys: Optional[List[str]] = None,
    keep_keys: Optional[List[str]] = None,
    return_optimal_values: bool = False
) -> pd.DataFrame:
    # We want to figure out what is the best achievable average loss IF we used optimal thresholds per subject
    sub_cols_to_keep = [
        sweep_key,
        y_key,
        'data_id',
    ]
    # Additional features we want to track.
    if keep_keys is not None:
        sub_cols_to_keep += keep_keys
    # Group keys are necessary for gettting opt temperatures
    if group_keys is not None:
        sub_cols_to_keep += group_keys
    else:
        group_keys = []
    # Filter out the columns we want to keep
    reduced_data_df = raw_data[sub_cols_to_keep].drop_duplicates().reset_index(drop=True)
    # Get the optimal temperature for each data_id
    optimal_df = reduced_data_df.loc[reduced_data_df.groupby(group_keys + ['data_id'])[y_key].idxmin()].reset_index(drop=True)
    # We want, per split, to get the average loss if we used the optimal temperature for each subject
    if group_keys is not None:
        grouped_opt_df = optimal_df.groupby(group_keys)
    else:
        grouped_opt_df = optimal_df
    # Mean across all the groups
    opt_values_df = grouped_opt_df.agg({y_key: 'mean'}).reset_index()
    if return_optimal_values:
        return opt_values_df, optimal_df
    else:
        # Return only the best values
        return opt_values_df 


