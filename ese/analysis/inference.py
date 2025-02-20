# torch imports
import torch
# ionpy imports
from ionpy.experiment.util import fix_seed
from ionpy.util.torchutils import to_device
# local imports
from .image_records import get_image_stats
from .checks import global_cal_sanity_check
import ese.analysis.analysis_utils.inference_utils as inf_utils
from .pixel_records import (
    update_toplabel_pixel_meters,
    update_cw_pixel_meters
)
from ..experiment.utils import (
    exp_patch_predict,
    show_inference_examples, 
)
# Misc imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from typing import Any, Optional
from pydantic import validate_arguments
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_inference(
    config: Any,
) -> None:

    # Get the config dictionary
    if not isinstance(config, dict):
        inference_cfg_dict = config.to_dict()
    else:
        inference_cfg_dict = config

    # Ensure that inference seed is the same.
    fix_seed(inference_cfg_dict['experiment']['inference_seed'])

    # Initialize all the objects needed for inference.
    inference_init_obj = inf_utils.cal_stats_init(inference_cfg_dict)
    inf_data_opts = inference_init_obj['dataobjs'].keys()

    # Loop through the data, gather your stats!
    if inference_cfg_dict["log"]["gether_inference_stats"]:
        tracker_objs = {
            "inf_cfg_dict": inference_cfg_dict,
            "inf_init_obj": inference_init_obj,
            "predictions": {}
        }
        # A dataloader is something that iterates through a set of datapoints we want to
        # run inference on. The invariant here is that we should expect to do inference
        # on every data point in the dataloader.
        for data_cfg_str in inf_data_opts:
            # Make the data opt args for this particular data configuration.
            if len(data_cfg_str) > 0:
                data_props = dict(item.split(':') for item in data_cfg_str.split('^'))
                data_props['data_cfg_str'] = data_cfg_str 
            else:
                data_props = {'data_cfg_str': data_cfg_str}
            # Iterate through this configuration's dataloader.
            if "support_sampler" in inference_init_obj['dataobjs'][data_cfg_str]:
                if inference_cfg_dict['experiment']['crosseval_incontex']:
                    crosseval_incontext_dataloader_loop(
                        inf_data_obj=inference_init_obj['dataobjs'][data_cfg_str],
                        data_props=data_props,
                        **tracker_objs
                    )
                else:
                    standard_incontext_dataloader_loop(
                        inf_data_obj=inference_init_obj['dataobjs'][data_cfg_str],
                        data_props=data_props,
                        **tracker_objs
                    )
            else:
                standard_dataloader_loop(
                    inf_data_obj=inference_init_obj['dataobjs'][data_cfg_str],
                    data_props=data_props,
                    **tracker_objs
                )
        # Save the records at the end too
        inf_utils.save_trackers(inference_init_obj["output_root"], trackers=inference_init_obj["trackers"])
        # Optionally, save the prediction logits.
        if inference_cfg_dict['log']["save_preds"]:
            inf_utils.save_preds(tracker_objs["predictions"], output_root=inference_init_obj["output_root"])

    # After the forward loop, we can calculate the global calibration metrics.
    if inference_cfg_dict["log"]["compute_global_metrics"]:
        # After the final pixel_meters have been saved, we can calculate the global calibration metrics and
        # insert them into the saved image_level_record dataframe.
        image_stats_dir = inference_init_obj["output_root"] / "image_stats.pkl"
        log_image_df = pd.read_pickle(image_stats_dir)
        # Loop through the calibration metrics and add them to the dataframe.
        for cal_metric_name, cal_metric_dict in inference_cfg_dict["global_cal_metrics"].items():
            # Add a dummy column to the dataframe.
            log_image_df[cal_metric_name] = np.nan
            # Iterate through the data opts, and replace the rows corresponding to the data opt with the cal losses.
            for data_cfg_str in inf_data_opts:
                assert cal_metric_dict['cal_type'] in ["classwise", "toplabel"],\
                    f"Calibration type {cal_metric_dict['cal_type']} not recognized."
                tracker_key = "cw_pixel_meter_dict" if cal_metric_dict['cal_type'] == "classwise" else "tl_pixel_meter_dict"
                # Calculate the loss.
                cal_loss = cal_metric_dict['_fn'](
                    pixel_meters_dict=inference_init_obj["trackers"][tracker_key][data_cfg_str]
                ).item() 
                # Replace the rows of log_image_df with column 'data_cfg_str' 
                log_image_df.loc[log_image_df["data_cfg_str"] == data_cfg_str, cal_metric_name] = cal_loss

        # Save the dataframe again.
        if inference_cfg_dict["log"]["log_pixel_stats"]:
            log_image_df.to_pickle(image_stats_dir)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_dataloader_loop(
    inf_cfg_dict,
    inf_init_obj,
    predictions,
    inf_data_obj, 
    data_props: Optional[dict] = {}
):
    dloader = inf_data_obj["dloader"]
    iter_dloader = iter(dloader)

    for batch_idx in range(len(dloader)):
        batch = next(iter_dloader)

        print(f"Working on batch #{batch_idx} out of",\
            len(dloader), "({:.2f}%)".format(batch_idx / len(dloader) * 100), end="\r")

        forward_batch = {
            "batch_idx": batch_idx,
            **data_props,
            **batch
        }
        # Final thing (just for our machinery), convert the data_id to a np.array.
        forward_batch["data_id"] = np.array(forward_batch["data_id"])
        # Place the output dir in the inf_cfg_dict
        inf_cfg_dict["output_root"] = inf_init_obj["output_root"]
        # Gather the forward item.
        forward_item = {
            "raw_batch": forward_batch,
            "inf_cfg_dict": inf_cfg_dict,
            "inf_init_obj": inf_init_obj,
            "predictions": predictions
        }
        # Run the forward loop
        standard_image_forward_loop(**forward_item)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_incontext_dataloader_loop(
    inf_cfg_dict,
    inf_init_obj,
    predictions,
    inf_data_obj, 
    data_props: Optional[dict] = {}
):
    num_supports = inf_cfg_dict['experiment']['num_supports']
    dloader = inf_data_obj["dloader"]

    for sup_idx in range(num_supports):
        # Sample the support set.
        sx, sy = inf_utils.sample_support(
            inf_cfg_dict,
            inf_data_obj,
            inf_init_obj,
            sup_idx=sup_idx
        )
        # We need to make a new iter for each support set.
        iter_dloader = iter(dloader)
        # Go through the dataloader.
        for batch_idx in range(len(dloader)):
            batch = next(iter_dloader)

            print(f"Working on batch #{batch_idx} out of",\
                len(dloader), "({:.2f}%)".format(batch_idx / len(dloader) * 100), end="\r")

            forward_batch = {
                "batch_idx": batch_idx,
                "support_images": sx,
                "support_labels": sy,
                "support_idx": sup_idx,
                **data_props,
                **batch
            }
            # Final thing (just for our machinery), convert the data_id to a np.array.
            forward_batch["data_id"] = np.array(forward_batch["data_id"])
            # Place the output dir in the inf_cfg_dict
            inf_cfg_dict["output_root"] = inf_init_obj["output_root"]
            # Gather the forward item.
            forward_item = {
                "raw_batch": forward_batch,
                "inf_cfg_dict": inf_cfg_dict,
                "inf_init_obj": inf_init_obj,
                "predictions": predictions
            }
            # Run the forward loop
            standard_image_forward_loop(**forward_item)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def crosseval_incontext_dataloader_loop(
    inf_cfg_dict,
    inf_init_obj,
    predictions,
    inf_data_obj, 
    data_props: Optional[dict] = {}
):
    assert inf_cfg_dict['dataloader']['batch_size'] == 1, "Batch size must be 1 for cross-evaluating in-context."
    dloader = inf_data_obj["dloader"]
    iter_dloader = iter(dloader)
    num_supports = inf_cfg_dict['experiment']['num_supports']
    # Go through the dataloader.
    for batch_idx in range(len(dloader)):
        batch = next(iter_dloader)
        print(f"Working on batch #{batch_idx} out of",\
            len(dloader), "({:.2f}%)".format(batch_idx / len(dloader) * 100), end="\r")
        # Loop through and sample multiple supports per subject.
        for sup_idx in range(num_supports):
            # Sample the support set.
            sx, sy = inf_utils.sample_support(
                inf_cfg_dict,
                inf_data_obj,
                inf_init_obj,
                batch=batch, # NOTE: This will ALWAYS exclude the current batch data id, even if in differnet splits.
                sup_idx=sup_idx
            )
            # Define the forward batch.
            forward_batch = {
                "batch_idx": batch_idx,
                "support_images": sx,
                "support_labels": sy,
                "support_idx": sup_idx,
                **data_props,
                **batch
            }
            # Final thing (just for our machinery), convert the data_id to a np.array.
            forward_batch["data_id"] = np.array(forward_batch["data_id"])
            # Place the output dir in the inf_cfg_dict
            inf_cfg_dict["output_root"] = inf_init_obj["output_root"]
            # Gather the forward item.
            forward_item = {
                "raw_batch": forward_batch,
                "inf_cfg_dict": inf_cfg_dict,
                "inf_init_obj": inf_init_obj,
                "predictions": predictions
            }
    
            # Run the forward loop
            standard_image_forward_loop(**forward_item)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_image_forward_loop(
    raw_batch,
    inf_cfg_dict,
    inf_init_obj,
    predictions
):
    # Get the experiment
    exp = inf_init_obj["exp"]
    
    # Get the data_ids that have enough label.
    valid_example_inds = inf_utils.filter_by_min_lab(
        label_map=raw_batch["label"], 
        min_pix=inf_cfg_dict["log"].get("min_fg_pixels", 0)
    )

    # If we have any valid indices, then we can proceed.
    if valid_example_inds.any():
        
        # Select batch inds by the valid indices.
        batch = inf_utils.select_batch_by_inds(
            batch=raw_batch, 
            valid_inds=valid_example_inds
        )

        # Get the example data
        image, label_map = batch.pop("img"), batch.pop("label")
        # Also try to pop the context images and labels if it exists.
        sx, sy = batch.pop("support_images", None), batch.pop("support_labels", None)
        incontext_mode = sx is not None and sy is not None
        # and put them on the device of our experiment.
        if image.device != exp.device:
            image, label_map = to_device((image, label_map), exp.device)
        
        # If there are inference kwargs, then we need to do a forward pass with those kwargs.
        inf_kwarg_grid = inf_utils.get_kwarg_sweep(inf_cfg_dict)
        # Iterate through each of the inference kwargs.
        for inf_kwarg_setting_dict in tqdm(inf_kwarg_grid, disable=(len(inf_kwarg_grid) == 1)):
            # Do a forward pass without gradients.
            with torch.no_grad():
                # If we have augs to apply on the image (on the GPU), then we need to do that here.
                if inf_init_obj.get('aug_pipeline', None): 
                    image = inf_init_obj['aug_pipeline'](image)

                # We can either perform the forward pass in one go, or do so in patches.
                patch_pred_kwargs = inf_kwarg_setting_dict.pop('patch_pred_kwargs', None)
                predict_kwargs = {
                    "x": image,
                    **inf_kwarg_setting_dict
                }

                # If we have support images then we need to add them to the predict_kwargs.
                if incontext_mode:
                    predict_kwargs.update({
                        "support_images": sx,
                        "support_labels": sy
                    })

                # If we are doing patch-based prediction then we need to do that here.
                if patch_pred_kwargs is not None:
                    exp_output = exp_patch_predict(exp, **patch_pred_kwargs, **predict_kwargs)
                else:
                    exp_output =  exp.predict(**predict_kwargs)

            # Get through all the batch elements.
            inference_batch_size = image.shape[0] 
            for batch_inference_idx in range(inference_batch_size):
                # For each of y_logits, y_probs, y_hard, we need to get the corresponding element.
                outputs_dict = {
                    tensor_type: out_tensor[batch_inference_idx, None, ...] for tensor_type, out_tensor in exp_output.items()
                }
                # Wrap the outputs into a dictionary.
                output_dict = {
                    "x": image[batch_inference_idx][None],
                    "y_true": label_map[batch_inference_idx][None],
                    **outputs_dict,
                    **inf_kwarg_setting_dict
                }
                # Some of our meta-data is also batched, and we need to idx it by the batch_inference_idx.
                for mdata_key, mdata_val in batch.items():
                    if isinstance(mdata_val, (torch.Tensor, np.ndarray)):
                        output_dict[mdata_key] = mdata_val[batch_inference_idx].item()
                    else:
                        output_dict[mdata_key] = mdata_val

                # If we are logging the predictions, then we need to do that here.
                if inf_cfg_dict['log']["save_preds"]:
                    predictions[output_dict['data_id']] = output_dict['y_logits'].cpu().numpy()
                
                ###########################
                # VISUALIZING IMAGE PREDS #
                ###########################
                if inf_cfg_dict["log"].get("show_examples", False):
                    show_inference_examples(output_dict, threshold=inf_kwarg_setting_dict.get("threshold", 0.5))

                # Get the calibration item info.  
                gather_output_dict_stats(
                    output_dict=output_dict,
                    inference_cfg=inf_cfg_dict,
                    trackers=inf_init_obj['trackers'],
                )

                # Save the records every so often, to get intermediate results. Note, because of data_ids
                # this can contain fewer than 'log interval' many items.
                if inf_init_obj['data_counter'] % inf_cfg_dict['log']['log_interval'] == 0:
                    inf_utils.save_trackers(inf_init_obj["output_root"], trackers=inf_init_obj['trackers'])

        # Increment the data counter.
        inf_init_obj['data_counter'] += 1


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def gather_output_dict_stats(
    output_dict,
    inference_cfg,
    trackers
):
    ################################################################
    # TEMPORARY ASSERT: WE ONLY HANDLE BINARY SEGMENTATION FOR NOW #
    ################################################################
    if "y_probs" in output_dict and output_dict["y_probs"] is not None:
        assert output_dict["y_probs"].shape[1] == 1, "Only binary segmentation is supported for now."
    if "y_logits" in output_dict and output_dict["y_logits"] is not None:
        assert output_dict["y_logits"].shape[1] == 1, "Only binary segmentation is supported for now."

    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    if "image_stats" in trackers:
        image_cal_metrics_dict = get_image_stats(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            image_stats=trackers["image_stats"]
        ) 

    #################################################################
    # CALIBRATION METRICS FOR THIS IMAGE (TOP-LABEL AND CLASS-WISE) #
    #################################################################
    # Top-label 
    if "tl_pixel_meter_dict" in trackers:
        image_tl_pixel_meter_dict = update_toplabel_pixel_meters(
            output_dict=output_dict,
            calibration_cfg=inference_cfg['global_calibration'],
            record_dict=trackers['tl_pixel_meter_dict'][output_dict['data_cfg_str']],
        )
    # Class-wise 
    if "cw_pixel_meter_dict" in trackers:
        image_cw_pixel_meter_dict = update_cw_pixel_meters(
            output_dict=output_dict,
            calibration_cfg=inference_cfg['global_calibration'],
            record_dict=trackers['cw_pixel_meter_dict'][output_dict['data_cfg_str']],
        )

    ##################################################################
    # SANITY CHECK THAT THE CALIBRATION METRICS AGREE FOR THIS IMAGE #
    ##################################################################
    if "image_stats" in trackers and\
        "tl_pixel_meter_dict" in trackers and\
         "cw_pixel_meter_dict" in trackers: 
        global_cal_sanity_check(
            data_id=output_dict["data_id"],
            slice_idx=output_dict["slice_idx"],
            inference_cfg=inference_cfg, 
            image_cal_metrics_dict=image_cal_metrics_dict, 
            image_tl_pixel_meter_dict=image_tl_pixel_meter_dict,
            image_cw_pixel_meter_dict=image_cw_pixel_meter_dict
        )
    