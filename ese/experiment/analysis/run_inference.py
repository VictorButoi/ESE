# torch imports
import torch
from torch import Tensor
import torch.nn.functional as F
# ionpy imports
from ionpy.util import Config
from ionpy.experiment.util import fix_seed
from ionpy.util.torchutils import to_device
# local imports
from .checks import global_cal_sanity_check
from .image_records import get_image_stats
from .analysis_utils.inference_utils import (
    aug_support,
    save_trackers,
    cal_stats_init
)
from .pixel_records import (
    update_toplabel_pixel_meters,
    update_cw_pixel_meters
)
from ..experiment.utils import (
    show_inference_examples, 
    reduce_ensemble_preds
)
# Misc imports
import math
import einops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Optional
from pydantic import validate_arguments
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_cal_stats(
    cfg: Config,
) -> None:
    # Get the config dictionary
    inference_cfg_dict = cfg.to_dict()

    # Ensure that inference seed is the same.
    fix_seed(inference_cfg_dict['experiment']['inference_seed'])

    # Initialize all the objects needed for inference.
    inference_init_obj = cal_stats_init(
        inference_cfg_dict, 
        yaml_cfg_dir="/storage/vbutoi/projects/ESE"
    )

    # Loop through the data, gather your stats!
    if inference_cfg_dict["log"]["gether_inference_stats"]:
        loop_base_args = {
            "inf_cfg_dict": inference_cfg_dict,
            "inf_init_obj": inference_init_obj,
        }
        for data_cfg_opt, cfg_dataloader_obj in inference_init_obj["dloaders"].items():
            # Make the data opt args for this particualr data configuration.
            data_props = dict(item.split(':') for item in data_cfg_opt.split('^'))
            data_props['data_cfg_opt'] = data_cfg_opt
            for label_idx, lab_dloader in cfg_dataloader_obj.items():
                data_props["label_idx"] = label_idx
                dloader_loop_args = {
                    "dloader": lab_dloader,
                    "data_props": data_props,
                    **loop_base_args
                }
                if inference_cfg_dict["model"]["_type"] == "incontext":
                    support_gen = inference_init_obj["supports"][data_cfg_opt][label_idx]
                    if inference_cfg_dict["experiment"]["fixed_support_sets"]:
                        for sup_idx in range(inference_cfg_dict['experiment']['supports_per_target']):
                            # Ensure that all subjects of the same label have the same support set.
                            rng = inference_cfg_dict['experiment']['inference_seed'] * (sup_idx + 1)
                            # Send the support set to the device
                            sx_cpu, sy_cpu, support_data_ids = support_gen[rng]
                            # Apply augmentation to the support set if defined.
                            if inference_init_obj['support_transforms'] is not None:
                                sx_cpu, sy_cpu = aug_support(sx_cpu, sy_cpu, inference_init_obj)
                            # if "Subject11" not in support_data_ids:
                            sx, sy = to_device((sx_cpu, sy_cpu), inference_init_obj["exp"].device)
                            # Run the dataloader loop with this particular support of images and labels.
                            dataloader_loop(
                                sup_idx=sup_idx,
                                support_dict={
                                    'images': sx[None], 
                                    'labels': sy[None], 
                                    'data_ids': support_data_ids
                                },
                                support_augs=inference_cfg_dict['experiment'].get('support_augs', None),
                                **dloader_loop_args,
                            )
                    else:
                        dataloader_loop(support_generator=support_gen, **dloader_loop_args)
                else:
                    dataloader_loop(**dloader_loop_args)
        # Save the records at the end too
        if inference_cfg_dict["log"]["log_image_stats"]:
            save_trackers(inference_init_obj["output_root"], trackers=inference_init_obj["trackers"])

    # After the forward loop, we can calculate the global calibration metrics.
    if inference_cfg_dict["log"]["summary_compute_global_metrics"]:
        # After the final pixel_meters have been saved, we can calculate the global calibration metrics and
        # insert them into the saved image_level_record dataframe.
        image_stats_dir = inference_init_obj["output_root"] / "image_stats.pkl"
        log_image_df = pd.read_pickle(image_stats_dir)
        # Loop through the calibration metrics and add them to the dataframe.
        for cal_metric_name, cal_metric_dict in inference_cfg_dict["global_cal_metrics"].items():
            # Add a dummy column to the dataframe.
            log_image_df[cal_metric_name] = np.nan
            # Iterate through the data opts, and replace the rows corresponding to the data opt with the cal losses.
            for data_cfg_opt in inference_init_obj["dloaders"].keys():
                assert cal_metric_dict['cal_type'] in ["classwise", "toplabel"],\
                    f"Calibration type {cal_metric_dict['cal_type']} not recognized."
                tracker_key = "cw_pixel_meter_dict" if cal_metric_dict['cal_type'] == "classwise" else "tl_pixel_meter_dict"
                # Calculate the loss.
                cal_loss = cal_metric_dict['_fn'](
                    pixel_meters_dict=inference_init_obj["trackers"][tracker_key][data_cfg_opt]
                ).item() 
                # Replace the rows of log_image_df with column 'data_cfg_opt' 
                log_image_df.loc[log_image_df["data_cfg_opt"] == data_cfg_opt, cal_metric_name] = cal_loss

        # Save the dataframe again.
        if inference_cfg_dict["log"]["log_pixel_stats"]:
            log_image_df.to_pickle(image_stats_dir)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dataloader_loop(
    inf_cfg_dict,
    inf_init_obj,
    dloader, 
    sup_idx: Optional[int] = None,
    data_props: Optional[dict] = {},
    support_augs: Optional[Any] = None,
    inference_type: Optional[str] = "standard",
):
    iter_dloader = iter(dloader)
    for batch_idx in range(len(dloader)):
        try:
            batch = next(iter_dloader)

            print(f"Working on batch #{batch_idx} out of",\
                len(dloader), "({:.2f}%)".format(batch_idx / len(dloader) * 100), end="\r")

            if isinstance(batch, list):
                batch = {
                    "img": batch[0],
                    "label": batch[1],
                    "data_id": batch[2],
                }
            forward_batch = {
                "batch_idx": batch_idx,
                **data_props,
                **batch
            }
            # Final thing (just for our machinery), convert the data_id to a np.array.
            forward_batch["data_id"] = np.array(forward_batch["data_id"])
            if inference_type == "incontext":
                forward_batch.update({
                    'support_augs': support_augs,
                    'sup_idx': sup_idx
                })
            # Place the output dir in the inf_cfg_dict
            inf_cfg_dict["output_root"] = inf_init_obj["output_root"]
            # Gather the forward item.
            forward_item = {
                "raw_batch": forward_batch,
                "inf_cfg_dict": inf_cfg_dict,
                "inf_init_obj": inf_init_obj,
            }

            # Run the forward loop
            input_type = inf_cfg_dict['data']['input_type']
            if input_type == 'volume':
                volume_forward_loop(**forward_item)
            elif input_type == 'image':
                standard_image_forward_loop(**forward_item)
            else:
                raise ValueError(f"Input type {input_type} not recognized.")

        except Exception as e:
            raise e


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    raw_batch,
    inf_cfg_dict,
    inf_init_obj
):
    # Get the batch info, these are B x V x H x W
    image_vol_cuda, label_vol_cuda = to_device((raw_batch.pop("img"), raw_batch.pop("label")), inf_init_obj["exp"].device)
    # Go through each slice and predict the metrics.
    slices_per_fp = inf_cfg_dict["experiment"].get("slices_per_fp", 1)
    # Num iterations is the number of slices divided by the number of slices per forward pass.
    num_iterations = math.ceil(image_vol_cuda.shape[1] / slices_per_fp)
    # Iterate through the slice chunks, making sure that on the last iteration we get the remaining slices.
    for slice_chunk in range(num_iterations):
        print(f"-> Working on slice chunk #{slice_chunk} out of", num_iterations,\
              "({:.2f}%)".format((slice_chunk/ num_iterations) * 100), end="\r")
        # Slice the image and label vols at the slice_chunk and move everything to the batch dimension.
        if slice_chunk == num_iterations - 1:
            slice_indices = torch.arange(slice_chunk*slices_per_fp, image_vol_cuda.shape[1])
        else:
            slice_indices = torch.arange(slice_chunk*slices_per_fp, (slice_chunk + 1)*slices_per_fp)
        # Slice the volumes by the chosen indices.
        image_slices = image_vol_cuda[:, slice_indices, ...]
        label_slices = label_vol_cuda[:, slice_indices, ...]
        # Move the slices to the batch dimension.
        image_batched_slices = einops.rearrange(image_slices, "b c h w -> (b c) 1 h w")
        label_batched_slices = einops.rearrange(label_slices, "b c h w -> (b c) 1 h w")
        # We need to expand the data_id in batch to account for the slices.
        expanded_slice_indices = slice_indices.repeat(len(raw_batch['data_id']))
        expanded_data_ids = []
        for data_id in raw_batch['data_id']:
            expanded_data_ids += [data_id] * len(slice_indices) 
        # Get the prediction with no gradient accumulation.
        slice_batch = raw_batch.copy()
        slice_batch.update({
            "img": image_batched_slices,
            "label": label_batched_slices,
            "data_id": np.array(expanded_data_ids),
            "slice_indices": expanded_slice_indices.cpu().numpy(),
        })
        standard_image_forward_loop(
            slice_batch,
            inf_cfg_dict=inf_cfg_dict,
            inf_init_obj=inf_init_obj,
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_image_forward_loop(
    raw_batch,
    inf_cfg_dict,
    inf_init_obj
):
    # Get the experiment
    exp = inf_init_obj["exp"]
    
    # Get the data_ids that have enough label.
    valid_example_inds = filter_by_min_lab(
        label_map=raw_batch["label"], 
        min_pix=inf_cfg_dict["log"].get("min_fg_pixels", 0)
    )

    # If we have any valid indices, then we can proceed.
    if valid_example_inds.any():
        
        # Select batch inds by the valid indices.
        batch = select_batch_by_inds(
            batch=raw_batch, 
            valid_inds=valid_example_inds
        )

        # Get the example data
        image = batch.pop("img")
        label_map = batch.pop("label")

        # Get your image label pair and define some regions.
        if image.device != exp.device:
            image, label_map = to_device((image, label_map), exp.device)
        
        # Label maps are soft labels so we need to convert them to hard labels.
        hard_lab_thresh = inf_cfg_dict["data"].get("label_threshold", None)
        if hard_lab_thresh is not None:
            label_map = (label_map > hard_lab_thresh).long()
        
        # Some args for the foward pass, only binary prediction is supported for now.
        predict_args = {
            'multi_class': False,
            'label': inf_cfg_dict['model'].get('pred_label', None)
        }
        if inf_cfg_dict["model"].get("ensemble", False):
            predict_args["combine_fn"] = "identity"
        
        # Optionally, we can resize the image at inference.
        resolution_cfg = inf_cfg_dict['experiment'].get('resolution', None) 

        if resolution_cfg: 
            input_res_cfg = resolution_cfg.get('input', None)
            if input_res_cfg:
                # Resize the image
                image = resize_image(input_res_cfg, image)

        # Do a forward pass.
        with torch.no_grad():
            exp_output =  exp.predict(image, **predict_args)

        # Go through the exp_output and see if they are None or not.
        for out_key, out_tensor in exp_output.items():
            # If the value is None, then drop the key.
            if out_tensor is None:
                exp_output.pop(out_key)
            # If we are resizing, then we need to resize the output.
            if resolution_cfg: 
                output_res_cfg = resolution_cfg.get('output', None)
                if output_res_cfg:
                    # Resize the image
                    exp_output[out_key] = resize_image(output_res_cfg, out_tensor)
                
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
                **outputs_dict
            }
            # Some of our meta-data is also batched, and we need to idx it by the batch_inference_idx.
            for mdata_key in batch.keys():
                mdata = batch[mdata_key]
                if isinstance(mdata, (torch.Tensor, np.ndarray)):
                    output_dict[mdata_key] = mdata[batch_inference_idx].item()
                else:
                    output_dict[mdata_key] = mdata

            # Get the calibration item info.  
            get_calibration_item_info(
                output_dict=output_dict,
                inference_cfg=inf_cfg_dict,
                trackers=inf_init_obj['trackers'],
            )

            # Save the records every so often, to get intermediate results. Note, because of data_ids
            # this can contain fewer than 'log interval' many items.
            if inf_cfg_dict["log"]["log_image_stats"] and (inf_init_obj['data_counter'] % inf_cfg_dict['log']['log_interval'] == 0):
                save_trackers(inf_init_obj["output_root"], trackers=inf_init_obj['trackers'])
            inf_init_obj['data_counter'] += 1


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
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

    ###########################
    # VISUALIZING IMAGE PREDS #
    ###########################
    if inference_cfg["log"].get("show_examples", False):
        show_inference_examples(
            output_dict, 
            inference_cfg=inference_cfg
        )

    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    if "image_stats" in trackers:
        image_cal_metrics_dict = get_image_stats(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            image_stats=trackers["image_stats"]
        ) 

    ###############################################################################################
    # If we are ensembling, then we need to reduce the predictions of the individual predictions. #
    ###############################################################################################
    if inference_cfg["model"].get("ensemble", False):
        # Get the reduced predictions
        ensembled_pred = reduce_ensemble_preds(
                            output_dict, 
                            inference_cfg=inference_cfg,
                        )
        output_dict.update(ensembled_pred)

    #################################################################
    # CALIBRATION METRICS FOR THIS IMAGE (TOP-LABEL AND CLASS-WISE) #
    #################################################################
    cal_args = {
        "output_dict": output_dict,
        "calibration_cfg": inference_cfg['global_calibration'],
    }
    # Top-label 
    if "tl_pixel_meter_dict" in trackers:
        image_tl_pixel_meter_dict = update_toplabel_pixel_meters(
            record_dict=trackers['tl_pixel_meter_dict'][output_dict['data_cfg_opt']],
            **cal_args
        )
    # Class-wise 
    if "cw_pixel_meter_dict" in trackers:
        image_cw_pixel_meter_dict = update_cw_pixel_meters(
            record_dict=trackers['cw_pixel_meter_dict'][output_dict['data_cfg_opt']],
            **cal_args
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
    

def resize_image(resize_cfg, image):
    # Get the original resolution of the image and write that we did resizing to the metadata.
    new_img_res = resize_cfg['size']
    # Resize the image
    interpolate_args = {
        "input": image, 
        "size": new_img_res, 
        "mode": resize_cfg['mode'], # this one is ok to be bilinear interpolation becuase this is what we trained on.
    }
    if resize_cfg['mode'] in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        interpolate_args['align_corners'] = resize_cfg['align_corners']
    # Resize the image or return if it's already the right size.
    if new_img_res != image.shape[-2:]:
        return torch.nn.functional.interpolate(**interpolate_args)
    else:
        return image


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def select_batch_by_inds(
    batch, 
    valid_inds
):
    subselect_batch = {}
    # We want to iterate through the keys, and if the key is a torch.Tensor or np.ndarray, then we want to select
    # the indices that are valid.
    for ikey in batch.keys():
        if isinstance(batch[ikey], (torch.Tensor, np.ndarray)):
            if len(valid_inds) == 1:
                if valid_inds[0]:
                    subselect_batch[ikey] = batch[ikey]
                else:
                    subselect_batch[ikey] = np.array([])
            else:
                subselect_batch[ikey] = batch[ikey][valid_inds]
        else:
            subselect_batch[ikey] = batch[ikey]
    # Return the filtered batch. 
    return subselect_batch 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def filter_by_min_lab(
    label_map, 
    min_pix: int = 0
):
    # If we have don't have a minimum number of foreground pixels, then we skip this image.
    # Because we allow for larger batchsizes, ie the label is B x 1 x H x W, we will get a vector of size B
    # where each element is the number of foreground pixels in the label map.
    valid_indices = torch.sum(label_map != 0, dim=(1, 2, 3)) >= min_pix 
    #  Return the valid indices.
    return valid_indices.cpu()