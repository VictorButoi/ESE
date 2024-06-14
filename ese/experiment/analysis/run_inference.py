# torch imports
import torch
# ionpy imports
from ionpy.util import Config
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
import numpy as np
import pandas as pd
from typing import Any, Optional
from pydantic import validate_arguments
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_cal_stats(
    cfg: Config,
) -> None:
    # Get the config dictionary
    inference_cfg_dict = cfg.to_dict()
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
                            rng = inference_cfg_dict['experiment']['seed'] * (sup_idx + 1)
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
    support_dict: Optional[dict] = None,
    support_generator: Optional[Any] = None,
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
            batch["data_id"] = batch["data_id"][0] # Works because batchsize = 1
            forward_batch = {
                "batch_idx": batch_idx,
                "sup_idx": sup_idx,
                "support_augs": support_augs,
                "data_props": data_props,
                **batch
            }
            # Place the output dir in the inf_cfg_dict
            inf_cfg_dict["output_root"] = inf_init_obj["output_root"]
            # Gather the forward item.
            forward_item = {
                "inf_cfg_dict": inf_cfg_dict,
                "inf_init_obj": inf_init_obj,
                "batch": forward_batch,
            }
            # Run the forward loop
            input_type = inf_cfg_dict['data']['input_type']
            if input_type == 'volume':
                volume_forward_loop(**forward_item)
            elif input_type == 'image':
                if inf_cfg_dict['model']['_type'] == 'incontext':
                    if support_dict is not None:
                        forward_item['support_dict'] = support_dict
                    elif support_generator is not None:
                        forward_item['support_generator'] = support_generator
                    else:
                        raise ValueError("Support set method not found.")
                    # Run the forward loop.
                    incontext_image_forward_loop(**forward_item)
                else:
                    standard_image_forward_loop(**forward_item)
            else:
                raise ValueError(f"Input type {input_type} not recognized.")
        except KeyError as k:
            print(k)
            print("Skipping this batch")
            pass
        except Exception as e:
            raise e


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    inf_cfg_dict,
    inf_init_obj,
    batch,
):
    # Get the batch info
    image_vol_cpu = batch.pop("img")
    label_vol_cpu = batch.pop("label")
    image_vol_cuda, label_vol_cuda = to_device((image_vol_cpu, label_vol_cpu), inf_init_obj["exp"].device)
    # Go through each slice and predict the metrics.
    num_slices = image_vol_cuda.shape[1]
    for slice_idx in range(num_slices):
        print(f"-> Working on slice #{slice_idx} out of", num_slices, "({:.2f}%)".format((slice_idx / num_slices) * 100), end="\r")
        # Get the prediction with no gradient accumulation.
        slice_batch = {
            "img": image_vol_cuda[:, slice_idx:slice_idx+1, ...],
            "label": label_vol_cuda[:, slice_idx:slice_idx+1, ...],
            **batch
        } 
        standard_image_forward_loop(
            inf_cfg_dict=inf_cfg_dict,
            inf_init_obj=inf_init_obj,
            batch=slice_batch,
            slice_idx=slice_idx,
        )
    assert slice_idx == num_slices - 1, "Slice index did not reach the end of the volume."


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_image_forward_loop(
    inf_cfg_dict,
    inf_init_obj,
    batch,
    slice_idx: Optional[int] = None,
):
    # Get the experiment
    exp = inf_init_obj["exp"]
    
    # Get the batch info
    image, label_map  = batch["img"], batch["label"]

    # If we have don't have a minimum number of foreground pixels, then we skip this image.
    if torch.sum(label_map != 0) >= inf_cfg_dict["log"].get("min_fg_pixels", 0):

        # Get your image label pair and define some regions.
        if image.device != exp.device:
            image, label_map = to_device((image, label_map), exp.device)
        
        # Label maps are soft labels so we need to convert them to hard labels.
        label_threshold = inf_cfg_dict["data"].get("label_threshold", None)
        if label_threshold is not None:
            label_map = (label_map > label_threshold).long()

        # Some args for the foward pass, only binary prediction is supported for now.
        predict_args = {
            'multi_class': False,
            'label': inf_cfg_dict['model'].get('pred_label', None)
        }
        if inf_cfg_dict["model"].get("ensemble", False):
            predict_args["combine_fn"] = "identity"

        # Do a forward pass.
        with torch.no_grad():
            exp_output =  exp.predict(image, **predict_args)

        # Wrap the outputs into a dictionary.
        output_dict = {
            "x": image,
            "y_true": label_map,
            "y_logits": exp_output.get("y_logits", None),
            "y_probs": exp_output.get("y_probs", None),
            "y_hard": exp_output.get("y_hard", None),
            "data_id": batch["data_id"], # Works because batchsize = 1
            "slice_idx": slice_idx,
            **batch["data_props"]
        }
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
def incontext_image_forward_loop(
    inf_cfg_dict,
    inf_init_obj,
    batch,
    trackers,
    support_dict: Optional[Any] = None,
    support_generator: Optional[Any] = None,
    slice_idx: Optional[int] = None,
):
    # Get the experiment object so we can make predictions with it.
    exp = inf_init_obj["exp"]
    # Get the batch info
    image = batch.pop("img")
    label_map = batch.pop("label")
    # If we have don't have a minimum number of foreground pixels, then we skip this image.
    if torch.sum(label_map != 0) >= inf_cfg_dict["log"].get("min_fg_pixels", 0):

        # Get your image label pair and define some regions.
        if image.device != exp.device:
            image, label_map = to_device((image, label_map), exp.device)
        
        # Label maps are soft labels so we need to convert them to hard labels.
        label_threshold = inf_cfg_dict["data"].get("label_threshold", None)
        if label_threshold is not None:
            label_map = (label_map > label_threshold).long()
       
        # Define the arguments that are shared by fixed supports and variable supports.
        common_forward_args = {
            "x": image,
            "y_true": label_map,
            "y_logits": None,
            "slice_idx": slice_idx,
            "data_id": batch['data_id'][0], # Works because batchsize = 1
            'support_augs': batch['support_augs'], # 'support_augs' is a dictionary of augmentations for the support set.
            **batch['data_props']
        }

        if support_dict is not None:
            assert inf_cfg_dict['ensemble']['num_members'] == 1, "Ensemble members must be 1 for fixed support sets."
            support_args = {
                "context_images": support_dict['images'],
                "context_labels": support_dict['labels'],
            }

            with torch.no_grad():
                if hasattr(exp, "predict"):
                    y_probs = exp.predict(
                        x=image, 
                        multi_class=False,
                        **support_args, 
                    )['y_probs']
                else:
                    y_logits = exp.model(target_image=image, **support_args)
                    if y_logits.shape[1] > 1:
                        y_probs = torch.softmax(y_logits, dim=1)
                    else:
                        y_probs = torch.sigmoid(y_logits)

            # Append the predictions to the ensemble predictions.
            y_hard = (y_probs > inf_cfg_dict['experiment']['threshold']).long() # (B, 1, H, W)
            y_probs = y_probs.unsqueeze(2) # B, 1, 1, H, W

            # Wrap the outputs into a dictionary.
            output_dict = {
                "y_hard": y_hard,
                "y_probs": y_probs,
                "sup_idx": batch['sup_idx'],
                "support_set": support_args,
                "support_data_ids": support_dict['data_ids'],
                **common_forward_args
            }
            # Get the calibration item info.  
            get_calibration_item_info(
                output_dict=output_dict,
                inference_cfg=inf_cfg_dict,
                trackers=trackers,
            )
        else:
            for sup_idx in range(inf_cfg_dict['experiment']['supports_per_target']):
                ensemble_probs_list = [] 
                ensemble_hards_list = []
                for ens_mem_idx in range(inf_cfg_dict['ensemble']['num_members']):
                    # Note: different subjects will use different support sets
                    # but different models will use the same support sets
                    rng = inf_cfg_dict['experiment']['seed'] * (sup_idx + 1) * (ens_mem_idx + 1) + batch['batch_idx'] 
                    sx_cpu, sy_cpu, _ = support_generator[rng]
                    # Apply augmentation to the support set if defined.
                    if inf_init_obj['support_transforms'] is not None:
                        sx_cpu, sy_cpu = aug_support(sx_cpu, sy_cpu, inf_init_obj)
                    sx, sy = to_device((sx_cpu, sy_cpu), exp.device)
                    # Package it into a dictionary.
                    support_args = {
                        "context_images": sx[None],
                        "context_labels": sy[None],
                    }
                    with torch.no_grad():
                        if hasattr(exp, "predict"):
                            y_probs = exp.predict(**support_args, x=image, multi_class=False)['y_probs']
                        else:
                            y_probs = torch.sigmoid(exp.model(**support_args, target_image=image))
                    # Append the predictions to the ensemble predictions.
                    ensemble_probs_list.append(y_probs) # (B, 1, H, W)
                    # Get the hard predictions.
                    ensemble_hards_list.append((y_probs > inf_cfg_dict['experiment']['threshold']).long()) # (B, 1, H, W)
                # Make the predictions (B, 2, E, H, W) by having the first channel be the background and second be the foreground.
                ensembled_probs = torch.stack(ensemble_probs_list, dim=0).permute(1, 2, 0, 3, 4) # (B, 1, E, H, W)
                ensembled_hard_pred = torch.cat(ensemble_hards_list, dim=1) # (B, E, H, W)
                # Wrap the outputs into a dictionary.
                output_dict = {
                    "sup_idx": sup_idx,
                    "y_probs": ensembled_probs,
                    "y_hard": ensembled_hard_pred,
                    "data_id": batch["data_id"][0], # Works because batchsize = 1
                    **common_forward_args
                }
                # Get the calibration item info.  
                get_calibration_item_info(
                    output_dict=output_dict,
                    inference_cfg=inf_cfg_dict,
                    trackers=trackers,
                )

        # Save the records every so often, to get intermediate results. Note, because of data_ids
        # this can contain fewer than 'log interval' many items.
        if inf_cfg_dict["log"]["log_image_stats"] and (inf_init_obj['data_counter'] % inf_cfg_dict['log']['log_interval'] == 0):
            save_trackers(inf_init_obj["output_root"], trackers=trackers)
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
    