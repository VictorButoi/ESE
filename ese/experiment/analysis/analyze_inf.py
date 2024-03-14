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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_cal_inference_stats(
    results_cfg: dict,
    load_cached: bool
) -> dict:
    # Build a dictionary to store the inference info.
    log_cfg = results_cfg["log"] 
    options_cfg = results_cfg["options"]

    log_root = log_cfg["root"] 
    results_cfg_hash = hash_dictionary(results_cfg)
    precomputed_results_path = log_root + "/results_cache/" + results_cfg_hash + ".pkl"
    skip_log_folders = ["wandb", "submitit", "binning_exp_logs"]
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
            for sub_exp in sub_exp_names:
                sub_exp_log_path = log_cfg["inference_group"] + "/" + sub_exp
                # Check to make sure this log wasn't the result of a crash.
                verify_graceful_exit(sub_exp_log_path, log_root=log_root)
                # Check to make sure that this log wasn't the result of a crash.
                all_inference_log_paths.append(sub_exp_log_path)
        # Add the inference paths if they exist.
        if "inference_paths" in log_cfg:
            for inf_path in log_cfg["inference_paths"]:
                all_inference_log_paths.append(inf_path)
        # Loop through every configuration in the log directory.
        skip_log_sets = []
        for log_path in all_inference_log_paths:
            log_dir = Path(os.path.join(log_root, log_path))
            for log_set in log_dir.iterdir():
                if log_set.name not in skip_log_folders:
                    try:
                        # Load the metadata file (json) and add it to the metadata dataframe.
                        config_dir = log_set / "config.yml"
                        with open(config_dir, 'r') as stream:
                            cfg_yaml = yaml.safe_load(stream)
                        cfg = HDict(cfg_yaml)
                        flat_cfg = valmap(list2tuple, cfg.flatten())
                        flat_cfg["log_set"] = log_set.name
                        # Count the number of ensemble members.
                        if "ensemble.member_paths" in flat_cfg and flat_cfg["ensemble.member_paths"] != "None":
                            flat_cfg["num_ensemble_members"] = len(flat_cfg["ensemble.member_paths"])
                        else:
                            flat_cfg["num_ensemble_members"] = "None"
                        # Remove some columns we don't care about.
                        for drop_key in [
                            "augmentations",
                            "dataset.augmentations",
                            "dataset.labels",
                            "qual_metrics", 
                            "image_cal_metrics", 
                            "global_cal_metrics", 
                            "calibration.bin_weightings", 
                            "calibration.conf_interval",
                            "model.filters",
                            "ensemble.member_paths"
                        ]:
                            if drop_key in flat_cfg:
                                flat_cfg.pop(drop_key)
                        # Convert the dictionary to a dataframe and concatenate it to the metadata dataframe.
                        cfg_df = pd.DataFrame(flat_cfg, index=[0])
                        metadata_df = pd.concat([metadata_df, cfg_df])
                    except Exception as e:
                        print(f"{e}. Skipping.")
                        skip_log_sets.append(log_set.name)
        # Gather the columns that have unique values amongst the different configurations.
        if options_cfg["remove_shared_columns"]:
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
            cal_metrics_dict=cal_metric_dict["global_cal_metrics"]
        )
        #############################
        inference_df = pd.DataFrame([])
        # Loop through every configuration in the log directory.
        for log_path in all_inference_log_paths:
            log_dir = Path(os.path.join(log_root, log_path))
            for log_set in log_dir.iterdir():
                if log_set.name not in skip_log_sets + skip_log_folders:
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
                        if options_cfg["load_pixel_meters"]:
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

        # Only choose rows with some minimal amount of foreground pixels.
        if options_cfg.get("min_fg_pixels", False):
            # Get the names of all columns that have "num_lab" in them.
            num_lab_cols = [col for col in inference_df.columns if "num_lab" in col]
            # Remove num_lab_0_pixels because it is background
            num_lab_cols.remove("num_lab_0_pixels")
            # Make a new column that is the sum of all the num_lab columns.
            inference_df['num_fg_pixels'] = inference_df[num_lab_cols].sum(axis=1)
            original_row_amount = len(inference_df)
            inference_df = inference_df[inference_df['num_fg_pixels'] >= options_cfg["min_fg_pixels"]]
            print(f"Dropping rows that don't meet minimum foreground pixel requirements. Dropped from {original_row_amount} -> {len(inference_df)} rows.")
        else:
            assert "WMH" not in results_cfg['log']['inference_group'],\
                "You must specify a min_fg_pixels value for WMH experiments." 

        # Drop the rows corresponding to NaNs in metric_score
        if options_cfg['drop_nan_metric_rows']:
            # Drop the rows where the metric score is NaN.
            original_row_amount = len(inference_df)
            # Get the triples of (data_idx, slice_idx, metric_name) where metric_score is NaN.
            unique_nan_triples = inference_df[inference_df['metric_score'].isna()][['data_id', 'slice_idx', 'image_metric']].drop_duplicates()
            # Drop the rows which match the triples.
            for _, row in unique_nan_triples.iterrows():
                inference_df = inference_df[
                    ~((inference_df['data_id'] == row['data_id']) & 
                      (inference_df['slice_idx'] == row['slice_idx']) & 
                      (inference_df['image_metric'] == row['image_metric']))
                      ]
            print(f"Dropping (datapoint, metric) pairs with NaN metric score. Dropped from {original_row_amount} -> {len(inference_df)} rows.")

        # Get the number of rows in image_info_df for each log set.
        num_rows_per_log_set = inference_df.groupby(["log.root", "log_set"]).size()
        if options_cfg["equal_rows_per_cfg_assert"]:
            # Make sure there is only one unique value in the above.
            assert len(num_rows_per_log_set.unique()) == 1, \
                f"The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}."
        else:
            if len(num_rows_per_log_set.unique()) != 1:
                print(f"Warning: The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}.")
        
        # Add new names for keys (helps with augment)
        inference_df["slice_idx"] = inference_df["slice_idx"].fillna("None")
        inference_df["model_class"] = inference_df["model._class"]
        inference_df["ensemble"] = inference_df["model.ensemble"]
        inference_df["pretrained_seed"] = inference_df["experiment.pretrained_seed"]

        # Go through several optional keys, and add them if they don't exist
        for optional_key in [
            "model._pretrained_class",
            "model.calibrator",
            "model.cal_stats_split",
            "ensemble.combine_fn",
            "ensemble.combine_quantity",
            "ensemble.member_w_metric",
            "groupavg_image_metric",
            "groupavg_metric_score"
        ]:
            new_key = optional_key.split(".")[-1]
            if optional_key in inference_df.columns:
                inference_df[new_key] = inference_df[optional_key].fillna("None")
            else:
                inference_df[new_key] = "None"
        
        if "model.normalize" in inference_df.columns:
            inference_df["model_norm"] = inference_df["model.normalize"].fillna("None")
        else:
            inference_df["model_norm"] = "None"

        if "ensemble.normalize" in inference_df.columns:
            inference_df["ensemble_norm"] = inference_df["ensemble.normalize"].fillna("None")
        else:
            inference_df["ensemble_norm"] = "None"

        # Here are a few common columns that we will always want in the dataframe.    
        def method_name(
            model_class, 
            _pretrained_class, 
            pretrained_seed, 
            ensemble, 
            combine_quantity, 
            combine_fn,
            ensemble_norm
        ):
            if ensemble:
                if ensemble_norm and not isinstance(ensemble_norm, str): 
                    return f"Ensemble ({combine_fn}, {combine_quantity}, norm)"
                else:
                    return f"Ensemble ({combine_fn}, {combine_quantity})" 
            else:
                if model_class in ["Vanilla", "FT_CE", "FT_Dice"]:
                    return f"UNet (seed={pretrained_seed})"
                elif _pretrained_class == "None":
                    return f"{model_class.split('.')[-1]} (seed={pretrained_seed})"
                else:
                    return f"{_pretrained_class.split('.')[-1]} (seed={pretrained_seed})"

        def calibrator(calibrator, model_class, model_norm, cal_stats_split):
            # Add the normalization to the calibrator name.
            if "Binning" in model_class:
                if model_norm:
                    calibrator += f" (norm,{cal_stats_split})"
                else:
                    calibrator += f" ({cal_stats_split})"
            return calibrator 

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
        if options_cfg['add_baseline_rows']:
            unet_avg = get_average_unet_baselines(
                inference_df, 
                num_seeds=4, # Used as a sanity check.
                group_metrics=list(cal_metrics.keys())
            )
            inference_df = pd.concat([inference_df, unet_avg], axis=0, ignore_index=True)

        # We want to add a bunch of new rows for Dice Loss that are the same as Dice but with a different metric score
        # that is 1 - metric_score.
        if options_cfg['add_dice_loss_rows']:
            inference_df = add_dice_loss_rows(inference_df, opts_cfg=options_cfg)

        # Save the inference info to a pickle file.
        with open(precomputed_results_path, 'wb') as f:
            pickle.dump(inference_df, f)
    else:
        # load the inference info from the pickle file.
        with open(precomputed_results_path, 'rb') as f:
            inference_df = pickle.load(f)

    # Get the number of rows in image_info_df for each log set.
    final_num_rows_per_log_set = inference_df.groupby(["log.root", "log_set"]).size()
    # Print information about each log set.
    print("Finished loading inference stats.")
    print(f"Log amounts: {final_num_rows_per_log_set}")

    # Finally, return the dictionary of inference info.
    return inference_df

