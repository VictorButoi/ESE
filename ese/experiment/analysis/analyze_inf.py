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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_cal_inference_stats(
    results_cfg: dict,
    load_cached: bool
) -> dict:
    # Build a dictionary to store the inference info.
    log_cfg = results_cfg["log"] 
    log_root = log_cfg["root"] 
    results_cfg_hash = hash_dictionary(results_cfg)
    precomputed_results_path = log_root + "/results_cache/" + results_cfg_hash + ".pkl"
    # Check to see if we have already built the inference info before.
    if not load_cached or not os.path.exists(precomputed_results_path):
        metadata_df = pd.DataFrame([])
        # Gather inference log paths.
        all_inference_log_paths = []
        if "inference_group" in log_cfg:
            sub_exp_names = os.listdir(log_root + "/" + log_cfg["inference_group"])
            # Remove the 'debug' or upper bounds folders if they exists.
            if "debug" in sub_exp_names:
                sub_exp_names.remove("debug")
            if "ensemble_upper_bounds" in sub_exp_names:
                sub_exp_names.remove("ensemble_upper_bounds")
            # Combine the inference group with the sub experiment names.
            all_inference_log_paths += [log_cfg["inference_group"] + "/" + sub_exp for sub_exp in sub_exp_names]
        if "inference_paths" in log_cfg:
            for inf_path in log_cfg["inference_paths"]:
                all_inference_log_paths.append(inf_path)
        # Loop through every configuration in the log directory.
        skip_log_sets = []
        for log_path in all_inference_log_paths:
            log_dir = Path(os.path.join(log_root, log_path))
            for log_set in log_dir.iterdir():
                if log_set.name not in ["wandb", "submitit"]:
                    try:
                        # Load the metadata file (json) and add it to the metadata dataframe.
                        config_dir = log_set / "config.yml"
                        with open(config_dir, 'r') as stream:
                            cfg_yaml = yaml.safe_load(stream)
                        cfg = HDict(cfg_yaml)
                        flat_cfg = valmap(list2tuple, cfg.flatten())
                        flat_cfg["log_set"] = log_set.name
                        # Remove some columns we don't care about.
                        for drop_key in [
                            "augmentations",
                            "dataset.augmentations",
                            "qual_metrics", 
                            "image_cal_metrics", 
                            "global_cal_metrics", 
                            "calibration.bin_weightings", 
                            "calibration.conf_interval",
                            "model.filters"
                            ]:
                            if drop_key in flat_cfg:
                                flat_cfg.pop(drop_key)
                        # If the column 'model.ensemble_cfg' is in the columns,
                        # we need to make two new columns for the combine function and quantity.
                        if 'model.ensemble_cfg' in flat_cfg.keys() and flat_cfg['model.ensemble_cfg'] is not None:
                            flat_cfg['model.ensemble_combine_fn'] = flat_cfg['model.ensemble_cfg'][0]
                            flat_cfg['model.ensemble_combine_quantity'] = flat_cfg['model.ensemble_cfg'][1]
                            flat_cfg.pop('model.ensemble_cfg')
                        # Convert the dictionary to a dataframe and concatenate it to the metadata dataframe.
                        cfg_df = pd.DataFrame(flat_cfg, index=[0])
                        metadata_df = pd.concat([metadata_df, cfg_df])
                    except Exception as e:
                        print(f"{e}. Skipping.")
                        skip_log_sets.append(log_set.name)
        # Gather the columns that have unique values amongst the different configurations.
        if log_cfg["remove_shared_columns"]:
            meta_cols = []
            for col in metadata_df.columns:
                if len(metadata_df[col].unique()) > 1:
                    meta_cols.append(col)
        else:
            meta_cols = metadata_df.columns
        ##################################
        # INITIALIZE CALIBRATION METRICS #
        ##################################
        cal_metric_dict = yaml.safe_load(open(results_cfg["calibration"]["metric_cfg_file"], 'r'))
        compute_cal_mets = results_cfg["log"].get("compute_cal_metrics", False)
        cal_metrics = preload_calibration_metrics(
            base_cal_cfg=results_cfg["calibration"],
            cal_metrics_dict=cal_metric_dict
        )
        #############################
        inference_df = pd.DataFrame([])
        # Loop through every configuration in the log directory.
        for log_path in all_inference_log_paths:
            log_dir = Path(os.path.join(log_root, log_path))
            for log_set in log_dir.iterdir():
                if log_set.name not in skip_log_sets +["wandb", "submitit"]:
                    try:
                        # Get the metadata corresponding to this log set.
                        metadata_log_df = metadata_df[metadata_df["log_set"] == log_set.name]
                        # Optionally load the information from image-based metrics.
                        log_image_df = pd.read_pickle(log_set / "image_stats.pkl")
                        log_image_df["log_set"] = log_set.name
                        # Add the columns from the metadata dataframe that have unique values.
                        for col in meta_cols:
                            assert len(metadata_log_df[col].unique()) == 1, \
                                f"Column {col} has more than one unique value in the metadata dataframe for log set {log_set}."
                            log_image_df[col] = metadata_log_df[col].values[0]
                        # Optionally load the pixel stats.
                        if log_cfg["load_pixel_meters"]:
                            with open(log_set / "pixel_stats.pkl", 'rb') as f:
                                pixel_meter_dict = pickle.load(f)
                            # Loop through the calibration metrics and add them to the dataframe.
                            for cal_metric_name, cal_metric_dict in cal_metrics.items():
                                if cal_metric_name not in log_image_df.columns and compute_cal_mets:
                                    log_image_df[cal_metric_name] = cal_metric_dict['_fn'](
                                        pixel_meters_dict=pixel_meter_dict
                                    ).item() 
                        # Add this log to the dataframe.
                        inference_df = pd.concat([inference_df, log_image_df])
                    except Exception as e:
                        print(f"Error loading image stats file for {log_set}. {e}. Skipping.")
        #########################################
        # POST-PROCESSING STEPS
        #########################################
        # Remove any final columns we don't want
        for drop_key in ["conf_interval"]:
            # If the key is in the dataframe, remove the column.
            if drop_key in inference_df.columns:
                inference_df = inference_df.drop(drop_key, axis=1)
        # Get the number of rows in image_info_df for each log set.
        num_rows_per_log_set = inference_df.groupby(["log.root", "log_set"]).size()
        
        if log_cfg["equal_rows_per_cfg_assert"]:
            # Make sure there is only one unique value in the above.
            assert len(num_rows_per_log_set.unique()) == 1, \
                f"The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}."
        else:
            if len(num_rows_per_log_set.unique()) != 1:
                print(f"Warning: The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}.")
        
        # Only choose rows with some minimal amount of foreground pixels.
        if "min_fg_pixels" in log_cfg:
            # Get the names of all columns that have "num_lab" in them.
            num_lab_cols = [col for col in inference_df.columns if "num_lab" in col]
            # Remove num_lab_0_pixels because it is background
            num_lab_cols.remove("num_lab_0_pixels")
            # Make a new column that is the sum of all the num_lab columns.
            inference_df['num_fg_pixels'] = inference_df[num_lab_cols].sum(axis=1)
            inference_df = inference_df[inference_df['num_fg_pixels'] >= log_cfg["min_fg_pixels"]]
        else:
            assert "WMH" not in results_cfg['log']['inference_group'],\
                "You must specify a min_fg_pixels value for WMH experiments." 
        
        # Add new names for keys (helps with augment)
        inference_df["slice_idx"] = inference_df["slice_idx"].fillna("None")
        inference_df["model_class"] = inference_df["model._class"]
        inference_df["ensemble"] = inference_df["model.ensemble"]
        inference_df["pretrained_seed"] = inference_df["experiment.pretrained_seed"]

        # Go through several optional keys, and add them if they don't exist
        for optional_key in [
            "model._pretrained_class",
            "model.ensemble_combine_fn",
            "model.ensemble_combine_quantity",
            "model.ensemble_w_metric",
            "groupavg_image_metric",
            "groupavg_metric_score"
        ]:
            new_key = optional_key.split(".")[-1]
            if optional_key in inference_df.columns:
                inference_df[new_key] = inference_df[optional_key].fillna("None")
            else:
                inference_df[new_key] = "None"

        # Here are a few common columns that we will always want in the dataframe.    
        def method_name(
            model_class, 
            _pretrained_class, 
            pretrained_seed, 
            ensemble, 
            ensemble_combine_quantity, 
            ensemble_combine_fn
        ):
            if ensemble:
                return f"Ensemble ({ensemble_combine_fn}, {ensemble_combine_quantity})" 
            else:
                if model_class == "Vanilla":
                    return f"UNet (seed={pretrained_seed})"
                elif _pretrained_class == "None":
                    return f"{model_class.split('.')[-1]} (seed={pretrained_seed})"
                else:
                    return f"{_pretrained_class.split('.')[-1]} (seed={pretrained_seed})"

        def calibrator(model_class):
            model_class_suffix = model_class.split('.')[-1]
            # Determine the calibration name.
            if "UNet" in model_class:
                return "Uncalibrated"
            elif model_class_suffix == "Identity":
                return "Vanilla"
            else:
                return model_class_suffix

        def joint_data_slice_id(data_id, slice_idx):
            return f"{data_id}_{slice_idx}"

        def configuration(method_name, calibrator):
            return f"{method_name}_{calibrator}"

        def model_type(ensemble):
            return 'group' if ensemble else 'individual'

        def metric_type(image_metric):
            if 'ECE' in image_metric or 'ELM' in image_metric:
                return 'calibration'
            else:
                return 'quality'

        def groupavg_image_metric(ensemble, groupavg_image_metric, image_metric):
            if ensemble:
                return groupavg_image_metric
            else:
                return f"GroupAvg_{image_metric}"

        def groupavg_metric_score(ensemble, groupavg_metric_score, metric_score):
            if ensemble:
                return groupavg_metric_score
            else:
                return metric_score

        # Add the new columns
        inference_df.augment(groupavg_image_metric)
        inference_df.augment(groupavg_metric_score)
        inference_df.augment(calibrator)
        inference_df.augment(metric_type)
        inference_df.augment(model_type)
        # Add some qol columns for ease of use.
        inference_df.augment(joint_data_slice_id)
        # Get the identifiers for our df.
        inference_df.augment(method_name)
        inference_df.augment(configuration)

        # Load the average unet baseline results.
        if log_cfg['add_baseline_rows']:
            unet_avg = get_average_unet_baselines(
                inference_df, 
                num_seeds=4, # Used as a sanity check.
                group_metrics=list(cal_metrics.keys())
            )
            inference_df = pd.concat([inference_df, unet_avg], axis=0, ignore_index=True)

        # Drop the rows corresponding to NaNs in metric_score
        if log_cfg['drop_nan_metric_rows']:
            # Drop the rows where the metric score is NaN.
            original_row_amount = len(inference_df)
            inference_df = inference_df.dropna(subset=['metric_score']).reset_index(drop=True)
            print(f"Dropping rows with NaN metric score. Dropped from {original_row_amount} -> {len(inference_df)} rows.")
        
        # We want to add a bunch of new rows for Dice Loss that are the same as Dice but with a different metric score
        # that is 1 - metric_score.
        if log_cfg['add_dice_loss_rows']:
            inference_df = add_dice_loss_rows(inference_df)
        
        # Print information about each log set.
        print("Finished loading inference stats.")
        print(f"Log amounts: {num_rows_per_log_set}")

        # Save the inference info to a pickle file.
        with open(precomputed_results_path, 'wb') as f:
            pickle.dump(inference_df, f)
    else:
        # load the inference info from the pickle file.
        with open(precomputed_results_path, 'rb') as f:
            inference_df = pickle.load(f)

    # Finally, return the dictionary of inference info.
    return inference_df

