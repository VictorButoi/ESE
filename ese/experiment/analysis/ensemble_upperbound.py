
# Misc imports
import os
from typing import Any, Optional
from pydantic import validate_arguments
# torch imports
import torch
from torch.nn import functional as F
# ionpy imports
from ionpy.util import Config
from ionpy.util.torchutils import to_device
# local imports
from .run_inference import save_records
from .analysis_utils.inference_utils import cal_stats_init
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_ensemble_ub(
    cfg: Config,
) -> None:
    # Get the config dictionary
    cfg_dict = cfg.to_dict()

    do_ensemble = cfg_dict["model"]["ensemble"]
    uncalibrated = not cfg_dict["model"]["calibrated"]
    assert do_ensemble and uncalibrated, "This function is only for uncalibrated ensembles."

    # Initialize the calibration statistics.
    cal_stats_components = cal_stats_init(cfg_dict)
    image_level_records = cal_stats_components["image_level_records"]
    image_stats_save_dir = cal_stats_components["image_level_dir"]
    
    # Loop through the data, gather your stats!
    with torch.no_grad():
        dataloader = cal_stats_components["dataloader"]
        for batch_idx, batch in enumerate(dataloader):
            print(f"Working on batch #{batch_idx} out of", len(dataloader), "({:.2f}%)".format(batch_idx / len(dataloader) * 100), end="\r")
            # Gather the forward item.
            forward_item = {
                "exp": cal_stats_components["inference_exp"],
                "batch": batch,
                "inference_cfg": cfg_dict,
                "image_level_records": image_level_records,
            }
            # Run the forward loop
            if cal_stats_components["input_type"] == "volume":
                volume_forward_loop(**forward_item)
            else:
                image_forward_loop(**forward_item)
            # Save the records every so often, to get intermediate results. Note, because of data_ids
            # this can contain fewer than 'log interval' many items.
            if batch_idx % cfg['log']['log_interval'] == 0:
                if image_level_records is not None:
                    save_records(image_level_records, image_stats_save_dir)
    # Save the records at the end too
    if image_level_records is not None:
        save_records(image_level_records, image_stats_save_dir)
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    exp: Any,
    batch: Any,
    inference_cfg: dict,
    image_level_records
):
    # Get the batch info
    image_vol_cpu, label_vol_cpu  = batch["img"], batch["label"]
    image_vol_cuda, label_vol_cuda = to_device((image_vol_cpu, label_vol_cpu), exp.device)
    # Go through each slice and predict the metrics.
    num_slices = image_vol_cuda.shape[1]
    for slice_idx in range(num_slices):
        print(f"-> Working on slice #{slice_idx} out of", num_slices, "({:.2f}%)".format((slice_idx / num_slices) * 100), end="\r")
        # Get the prediction with no gradient accumulation.
        slice_batch = {
            "img": image_vol_cuda[:, slice_idx:slice_idx+1, ...],
            "label": label_vol_cuda[:, slice_idx:slice_idx+1, ...],
            "data_id": batch["data_id"],
        } 
        image_forward_loop(
            exp=exp,
            batch=slice_batch,
            inference_cfg=inference_cfg,
            slice_idx=slice_idx,
            image_level_records=image_level_records,
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    exp: Any,
    batch: Any,
    inference_cfg: dict,
    image_level_records,
    slice_idx: Optional[int] = None,
):
    # Get the batch info
    image, label_map  = batch["img"], batch["label"]
    # Get your image label pair and define some regions.
    if image.device != exp.device:
        image, label_map = to_device((image, label_map), exp.device)
    # Do a forward pass.
    with torch.no_grad():
        exp_output =  exp.predict(
            image,
            multi_class=True,
            combine_fn="identity"
        )
    # Wrap the outputs into a dictionary.
    output_dict = {
        "x": image,
        "y_true": label_map.long(),
        "y_pred": exp_output["y_pred"],
        "y_hard": exp_output["y_hard"],
        "data_id": batch["data_id"][0], # Works because batchsize = 1
        "slice_idx": slice_idx,
        "ens_weights": exp.ens_mem_weights
    }
    # Get the calibration item info.  
    get_upperbound_info(
        output_dict=output_dict,
        inference_cfg=inference_cfg,
        image_level_records=image_level_records
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_upperbound_info(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records
):
    # Get the image level statistics.
    get_upper_bound_metric_stats(
        output_dict=output_dict,
        inference_cfg=inference_cfg,
        image_level_records=image_level_records
    ) 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_upper_bound_metric_stats(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records
):
    # Gather the individual predictions
    B, C, E, H, W = output_dict["y_pred"].shape
    ensemble_probs = torch.softmax(output_dict["y_pred"], dim=1) # B x C x E x H x W
    ensemble_hard_preds = torch.argmax(ensemble_probs, dim=1) # B x E x H x W
    # Get the upper bound prediction by going through and updating the prediction
    # by the pixels each model got right.
    ensemble_ub_pred = ensemble_probs[:, :, 0, ...] # B x C x H x W
    for ens_idx in range(E):
        correct_positions = (ensemble_hard_preds[:, ens_idx:ens_idx+1, ...] == output_dict["y_true"]) # B x 1 x H x W
        correct_index = correct_positions.repeat(1, C, 1, 1) # B x C x H x W
        ensemble_ub_pred[correct_index] = ensemble_probs[:, :, ens_idx, ...][correct_index]
    # Here we need a check that if we sum across channels we get 1.
    assert torch.allclose(ensemble_ub_pred.sum(dim=1), torch.ones((B, H, W), device=ensemble_ub_pred.device)),\
        "The upper bound prediction does not sum to 1 across channels."

    #############################################################
    # CALCULATE QUALITY METRICS
    #############################################################
    qual_metric_scores_dict = {}
    for qual_metric_name, qual_metric_dict in inference_cfg["qual_metrics"].items():
        qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](
            y_pred=ensemble_ub_pred,
            y_true=output_dict["y_true"],
            from_logits=False
        ).item() 

    # Calculate the amount of present ground-truth there is in the image per label.
    if inference_cfg["log"]["track_label_amounts"]:
        num_classes = output_dict["y_pred"].shape[1]
        y_true_one_hot = F.one_hot(output_dict["y_true"], num_classes=num_classes) # B x 1 x H x W x C
        label_amounts = y_true_one_hot.sum(dim=(0, 1, 2, 3)) # C
        label_amounts_dict = {f"num_lab_{i}_pixels": label_amounts[i].item() for i in range(num_classes)}
    
    for met_name in list(qual_metric_scores_dict.keys()):
        # Add the dataset info to the record
        record = {
            "image_metric": met_name,
            "metric_score": qual_metric_scores_dict[met_name],
            "data_id": output_dict["data_id"],
            "slice_idx": output_dict["slice_idx"],
        }
        if inference_cfg["log"]["track_label_amounts"]:
            record = {**record, **label_amounts_dict}
        # Add the record to the list.
        image_level_records.append(record)