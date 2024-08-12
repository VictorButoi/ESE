# misc imports
import os
import yaml
import json
import pickle
import hashlib
import pandas as pd
from pathlib import Path
from pydantic import validate_arguments
# ionpy imports
from ionpy.util.config import HDict, valmap
# local imports
from .analysis_utils.inference_utils import (
    add_dice_loss_rows,
    verify_graceful_exit,
    get_average_unet_baselines,
    preload_calibration_metrics,
)


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def hash_dictionary(dictionary):
    # Convert the dictionary to a JSON string
    json_str = json.dumps(dictionary, sort_keys=True)
    # Create a hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the JSON string encoded as bytes
    hash_object.update(json_str.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


def hash_list(input_list):
    # Convert the list to a JSON string
    json_str = json.dumps(input_list, sort_keys=True)
    # Create a hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the JSON string encoded as bytes
    hash_object.update(json_str.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_pixel_meters(
    log_set_dir: Path,
    results_cfg: dict
):
    cal_metric_dict = yaml.safe_load(open(results_cfg["calibration"]["metric_cfg_file"], 'r'))
    cal_metrics = preload_calibration_metrics(
        base_cal_cfg=results_cfg["calibration"],
        cal_metrics_dict=cal_metric_dict["global_cal_metrics"]
    )

    with open(log_set_dir / "pixel_stats.pkl", 'rb') as f:
        pixel_meter_dict = pickle.load(f)

    # Loop through the calibration metrics and add them to the dataframe.
    global_met_df = {}
    for cal_metric_name, cal_metric_dict in cal_metrics.items():
        global_met_df[cal_metric_name] = cal_metric_dict['_fn'](
            pixel_meters_dict=pixel_meter_dict
        ).item() 
    return global_met_df


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_flat_cfg(
    cfg_name: str, 
    cfg_dir: Path
):
    with open(cfg_dir, 'r') as stream:
        logset_cfg_yaml = yaml.safe_load(stream)
    logset_cfg = HDict(logset_cfg_yaml)
    logset_flat_cfg = valmap(list2tuple, logset_cfg.flatten())
    logset_flat_cfg["log_set"] = cfg_name
    # For the rest of the keys, if the length of the value is more than 1, convert it to a string.
    for key in logset_flat_cfg:
        if isinstance(logset_flat_cfg[key], list) or isinstance(logset_flat_cfg[key], tuple):
            logset_flat_cfg[key] = str(logset_flat_cfg[key])
    # Return the flattened configuration.
    return logset_flat_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_cal_inference_stats(
    results_cfg: dict,
    load_cached: bool
) -> dict:
    # Build a dictionary to store the inference info.
    log_cfg = results_cfg["log"] 
    log_root = log_cfg["root"] 

    # Get the hash of the results_cfg dictionary.
    results_cfg_hash = hash_dictionary(results_cfg)
    precomputed_results_path = log_root + "/results_cache/" + results_cfg_hash + ".pkl"

    # Check to see if we have already built the inference info before.
    if not load_cached or not os.path.exists(precomputed_results_path):
        skip_log_folders = [
            "wandb", 
            "submitit", 
            "binning_exp_logs", 
            "ensemble_upper_bounds",
            "debug"
        ]
        # Gather inference log paths.
        all_inference_log_paths = []
        for inf_group in log_cfg["inference_groups"]:
            inf_group_dir = log_root + "/" + inf_group
            group_folders = os.listdir(inf_group_dir)
            # If 'submitit' is in the highest dir, then we don't have subdirs (new folder structure).
            if "submitit" in group_folders:
                # Check to make sure this log wasn't the result of a crash.
                verify_graceful_exit(inf_group_dir, log_root=log_root)
                # Check to make sure that this log wasn't the result of a crash.
                all_inference_log_paths.append(inf_group_dir)
            # Otherwise, we had separated our runs in 'log sets', which isn't a good level of abstraction.
            # but it's what we had done before.
            else:
                for sub_exp in group_folders:
                    sub_exp_log_path = inf_group + "/" + sub_exp
                    # TODO: FIX THIS, I HATE THE SKIP_LOG_FOLDER PARADIGM.
                    # Verify that it is a folder and also that it is not in the skip_log_folders.
                    if os.path.isdir(sub_exp_log_path) and sub_exp not in skip_log_folders:
                        # Check to make sure this log wasn't the result of a crash.
                        verify_graceful_exit(sub_exp_log_path, log_root=log_root)
                        # Check to make sure that this log wasn't the result of a crash.
                        all_inference_log_paths.append(sub_exp_log_path)
        # Loop through every configuration in the log directory.
        metadata_pd_collection = []
        for log_path in all_inference_log_paths:
            log_dir = Path(os.path.join(log_root, log_path))
            # In this case, there are multiple log sets in the log directory.
            for log_set in log_dir.iterdir():
                # TODO: FIX THIS, I HATE THE SKIP_LOG_FOLDER PARADIGM.
                # Verify that log_set is a directory and that it's not in the skip_log_folders.
                if log_set.is_dir() and log_set.name not in skip_log_folders:
                    # Load the metadata file (json) and add it to the metadata dataframe.
                    logset_config_dir = log_set / "config.yml"
                    logset_flat_cfg = get_flat_cfg(cfg_name=log_set.name, cfg_dir=logset_config_dir)
                    # If there was a pretraining class, then we additionally add its config.
                    if results_cfg["options"].get('load_pretrained_cfg', True)\
                        and 'train.pretrained_dir' in logset_flat_cfg:
                        pretrained_cfg_dir = Path(logset_flat_cfg['train.pretrained_dir']) / "config.yml"
                        pt_flat_cfg = get_flat_cfg(cfg_name=log_set.name, cfg_dir=pretrained_cfg_dir)
                        # Add 'pretraining' to the keys of the pretrained config.
                        pt_flat_cfg = {f"pretraining_{key}": val for key, val in pt_flat_cfg.items()}
                        # Update the logset_flat_cfg with the pretrained config.
                        logset_flat_cfg.update(pt_flat_cfg)
                    # Append the df of the dictionary.
                    metadata_pd_collection.append(logset_flat_cfg)
        # Finally, concatenate all of the metadata dataframes.
        metadata_df = pd.DataFrame(metadata_pd_collection) 
        # Gather the columns that have unique values amongst the different configurations.
        if results_cfg["options"].get("remove_shared_columns", False):
            meta_cols = []
            for col in metadata_df.columns:
                if len(metadata_df[col].unique()) > 1:
                    meta_cols.append(col)
        else:
            meta_cols = metadata_df.columns
        #############################
        inference_pd_collection = []
        # Loop through every configuration in the log directory.
        for log_path in all_inference_log_paths:
            log_dir = Path(os.path.join(log_root, log_path))
            for log_set_path in log_dir.iterdir():
                # TODO: FIX THIS, I HATE THE SKIP_LOG_FOLDER PARADIGM.
                if log_set_path.is_dir() and log_set_path.name not in skip_log_folders:
                    # Optionally load the information from image-based metrics.
                    log_image_df = pd.read_pickle(log_set_path / "image_stats.pkl")
                    # Get the metadata corresponding to this log set.
                    metadata_log_df = metadata_df[metadata_df["log_set"] == log_set_path.name]
                    assert len(metadata_log_df) == 1, \
                        f"Metadata configuration must have one instance, found {len(metadata_log_df)}."
                    # Tile the metadata df the number of times to match the number of rows in the log_image_df.
                    tiled_metadata_log_df = pd.concat([metadata_log_df] * len(log_image_df), ignore_index=True)
                    # Add the columns from the metadata dataframe that have unique values.
                    logset_complete_df = pd.concat([log_image_df, tiled_metadata_log_df], axis=1)
                    # Optionally load the pixel stats.
                    if results_cfg["options"].get("get_glocal_cal_metrics", False):
                        logset_complete_df = pd.concat([
                            logset_complete_df, 
                            load_pixel_meters(log_set_path, results_cfg=results_cfg)
                        ], axis=1)
                    # Add this log to the dataframe.
                    inference_pd_collection.append(logset_complete_df)
        # Finally concatenate all of the inference dataframes.
        inference_df = pd.concat(inference_pd_collection, axis=0)

        #########################################
        # POST-PROCESSING STEPS
        #########################################
        # If slice_idx isn't in the columns, add it.
        if "slice_idx" not in inference_df.columns:
            inference_df["slice_idx"] = "None"
        # Drop the rows corresponding to NaNs in metric_score
        if results_cfg["options"].get('drop_nan_metric_rows', True):
            # Get the triples of (data_idx, slice_idx, metric_name) where metric_score is NaN.
            unique_nan_triples = inference_df[inference_df['metric_score'].isna()][['data_id', 'slice_idx', 'image_metric']].drop_duplicates()
            # Drop the rows which match the triples.
            for _, row in unique_nan_triples.iterrows():
                inference_df = inference_df[
                    ~((inference_df['data_id'] == row['data_id']) & 
                      (inference_df['slice_idx'] == row['slice_idx']) & 
                      (inference_df['image_metric'] == row['image_metric']))
                      ]
            print(f"Dropping (datapoint, metric) pairs with NaN metric score. Dropped from {len(inference_df)} -> {len(inference_df)} rows.")

        # Get the number of rows in image_info_df for each log set.
        num_rows_per_log_set = inference_df.groupby(["log.root", "log_set"]).size()
        if results_cfg["options"]["equal_rows_per_cfg_assert"]:
            # Make sure there is only one unique value in the above.
            assert len(num_rows_per_log_set.unique()) == 1, \
                f"The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}."
        else:
            if len(num_rows_per_log_set.unique()) != 1:
                print(f"Warning: The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}.")

        # Go through several optional keys, and add them if they don't exist
        new_columns = {}
        old_raw_keys = []
        # Go through several optional keys, and add them if they don't exist
        for raw_key in inference_df.columns:
            key_parts = raw_key.split(".")
            last_part = key_parts[-1]
            if last_part in ['_class', '_name']:
                new_key = "".join(key_parts)
            else:
                new_key = "_".join(key_parts)
            # If the new key isn't the same as the old key, add the new key.
            if new_key != raw_key and new_key not in inference_df.columns:
                new_columns[new_key] = inference_df[raw_key].fillna("None") # Fill the key with "None" if it is NaN.
                old_raw_keys.append(raw_key)

        # Add new columns to the DataFrame all at once
        inference_df = pd.concat([inference_df, pd.DataFrame(new_columns)], axis=1)
        # Delete the old keys
        inference_df.drop(columns=[col for col in inference_df.columns if col in old_raw_keys], inplace=True)

        # We want to add a bunch of new rows for Dice Loss that are the same as Dice but with a different metric score
        # that is 1 - metric_score.
        if results_cfg["options"].get('add_dice_loss_rows', True):
            inference_df = add_dice_loss_rows(inference_df, opts_cfg=results_cfg["options"])

        # If precomputed_results_path doesn't exist, create it.
        if not os.path.exists(os.path.dirname(precomputed_results_path)):
            os.makedirs(os.path.dirname(precomputed_results_path))
        # Save the inference info to a pickle file.
        with open(precomputed_results_path, 'wb') as f:
            pickle.dump(inference_df, f)
    else:
        # load the inference info from the pickle file.
        with open(precomputed_results_path, 'rb') as f:
            inference_df = pickle.load(f)

    # Get the number of rows in image_info_df for each log set.
    final_num_rows_per_log_set = inference_df.groupby(["log_root", "log_set"]).size()
    # Print information about each log set.
    print("Finished loading inference stats.")
    print(f"Log amounts: {final_num_rows_per_log_set}")

    # Prepare the final object
    inference_obj = {
        "df": inference_df,
        "preds": None,
        "dataset": None
    }

    # Get some arguments.
    load_preds = results_cfg["options"].get("load_predictions", False)
    load_inf_dataset = results_cfg["options"].get("load_inference_dataset", False)

    if load_preds:
        pass

    if load_inf_dataset:
        raise NotImplementedError("Loading the inference dataset is not yet implemented.")
    
    # Finally, return the dictionary of inference info.
    if not load_preds and not load_inf_dataset:
        return inference_df
    else:
        return inference_obj 

