# Torch imports
import torch
from torch.nn import functional as F
# Local imports
from ..metrics.local_ps import bin_stats_init
from ..experiment.utils import reduce_ensemble_preds
from ..utils.general import save_dict 
# Misc imports
from pydantic import validate_arguments
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    output_dict, 
    inference_cfg, 
    image_level_records
):
    # Define some common bin stat args
    bin_stat_args = {
        "y_true": output_dict["y_true"],
        "from_logits": False,
        "num_prob_bins": inference_cfg["local_calibration"]["num_prob_bins"],
        "neighborhood_width": inference_cfg["local_calibration"]["neighborhood_width"]
    }
    # (Both individual and reduced)
    if inference_cfg["model"]["ensemble"]:
        # Gather the individual predictions
        if output_dict['y_logits'] is not None and output_dict['y_probs'] is None:
            output_dict['y_probs'] = torch.softmax(output_dict['y_logits'], dim=1)
        # Get the individual predictions.
        ensemble_member_preds = [
            output_dict['y_probs'][:, :, ens_mem_idx, ...]\
            for ens_mem_idx in range(output_dict['y_probs'].shape[2])
        ]
        # Construct the input cfgs used for calulating metrics.
        ens_member_qual_input_cfgs = [
            {
                "y_pred": member_pred, 
                "y_true": output_dict["y_true"],
                "from_logits": False,
            } for member_pred in ensemble_member_preds
        ]
        ens_member_cal_input_cfgs = [
            {
                "y_pred": member_pred, 
                "y_true": output_dict["y_true"],
                "from_logits": False,
                "preloaded_obj_dict": bin_stats_init(member_pred, **bin_stat_args)
            } for member_pred in ensemble_member_preds
        ]
        # Get the reduced predictions
        qual_input_config = {
            'y_pred': reduce_ensemble_preds(
                        output_dict, 
                        inference_cfg=inference_cfg,
                    )['y_probs'],
            'y_true': output_dict['y_true'],
            'threshold': inference_cfg['experiment']['threshold'],
        }
        # Get the reduced predictions
        cal_input_config = {
            "preloaded_obj_dict": bin_stats_init(qual_input_config['y_pred'], **bin_stat_args),
            **qual_input_config
        }
    else:
        qual_input_config = {
            "y_pred": output_dict["y_probs"], # either (B, C, H, W) or (B, C, E, H, W), if ensembling
            "y_true": output_dict["y_true"], # (B, C, H, W)
            'threshold': inference_cfg['experiment']['threshold'],
        }
        cal_input_config = {
            **qual_input_config,
            "preloaded_obj_dict": bin_stats_init(qual_input_config['y_pred'], **bin_stat_args)
        }
    # Dicts for storing ensemble scores.
    grouped_scores_dict = {
        "calibration": {},
        "quality": {}
    }

    #############################################################
    # CALCULATE QUALITY METRICS
    #############################################################
    qual_metric_scores_dict = {}
    for qual_metric_name, qual_metric_dict in inference_cfg["qual_metrics"].items():
        # If we are ensembling, then we need to go through eahc member of the ensemble and calculate individual metrics
        # so we can get group averages.
        if inference_cfg["model"]["ensemble"]:
            # First gather the quality scores per ensemble member.
            #######################################################
            if inference_cfg["log"]["track_ensemble_member_scores"]:
                individual_qual_scores = []
                for ens_mem_input_cfg in ens_member_qual_input_cfgs:
                    member_qual_score = qual_metric_dict['_fn'](**ens_mem_input_cfg)
                    individual_qual_scores.append(member_qual_score)
                # Now place it in the dictionary.
                grouped_scores_dict['quality'][qual_metric_name] = torch.mean(torch.Tensor(individual_qual_scores))
            else:
                grouped_scores_dict['quality'][qual_metric_name] = None
            # Now get the ensemble quality score.
        qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**qual_input_config)
        # If you're showing the predictions, also print the scores.
        if inference_cfg["log"].get("show_examples", False):
            print(f"{qual_metric_name}: {qual_metric_scores_dict[qual_metric_name]}")

    #############################################################
    # CALCULATE CALIBRATION METRICS
    #############################################################
    cal_metric_errors_dict = {}
    for cal_metric_name, cal_metric_dict in inference_cfg["image_cal_metrics"].items():
        # If we are ensembling, then we need to go through eahc member of the ensemble and calculate individual metrics
        # so we can get group averages.
        if inference_cfg["model"]["ensemble"]:
            # First gather the calibration scores per ensemble member.
            #######################################################
            if inference_cfg["log"]["track_ensemble_member_scores"]:
                individual_cal_scores = []
                for ens_mem_input_cfg in ens_member_cal_input_cfgs:
                    member_cal_score = cal_metric_dict['_fn'](**ens_mem_input_cfg)
                    individual_cal_scores.append(member_cal_score)
                # Now place it in the dictionary.
                grouped_scores_dict['calibration'][cal_metric_name] = torch.mean(torch.Tensor(individual_cal_scores))
            else:
                grouped_scores_dict['calibration'][cal_metric_name] = None
        # Get the calibration error. 
        cal_metric_errors_dict[cal_metric_name] = cal_metric_dict['_fn'](**cal_input_config)
        # If you're showing the predictions, also print the scores.
        if inference_cfg["log"].get("show_examples", False):
            print(f"{cal_metric_name}: {cal_metric_errors_dict[cal_metric_name]}")
    
    assert not (len(qual_metric_scores_dict) == 0 and len(cal_metric_errors_dict) == 0), \
        "No metrics were specified in the config file."

    # If the label_idx is defined, we are dealing with a binary problem and we want to get the amount of pixels.
    volume_dict = {}
    # Get groundtruth label amount
    gt_volume = output_dict['y_true'].sum().item()
    # Get the soft prediction volume, this only works for binary problems.
    pred_volume = output_dict['y_probs'][:, 1, ...].sum().item()
    # Get thresholded prediction volume
    hard_volume = output_dict['y_hard'].sum().item()
    # Define a dictionary of the volumes.
    volume_dict.update({
        "gt_volume": gt_volume,
        "soft_volume": pred_volume,
        "hard_volume": hard_volume
    })
    
    # We wants to remove the keys corresponding to the image data.
    exclude_keys = [
        'x', 
        'y_true', 
        'y_logits', 
        'y_probs', 
        'y_hard', 
        'support_set'
    ]
    output_metadata = {k: v for k, v in output_dict.items() if k not in exclude_keys}

    # We want to generate a hash of this metadata, useful for accesing saved predictions.
    pred_hash = hash(str(output_metadata))
    # Iterate through all of the collected metrics and add them to the records.
    for met_name, met_score in {**qual_metric_scores_dict, **cal_metric_errors_dict}.items():
        if met_score is not None:
            met_score = met_score.item()
        # Add the dataset info to the record
        record = {
            "pred_hash": pred_hash,
            "image_metric": met_name,
            "metric_score": met_score,
            **output_metadata,
            **volume_dict,
        }
        # Add the record to the list.
        image_level_records.append(record)

    # If we are logging the predictions, then we need to do that here.
    if inference_cfg['log'].get("save_preds", False):
        # Make a copy of the output dict.
        output_dict_copy = output_dict.copy()
        remove_keys = [
            'x', 
            'y_true', 
            'y_hard', 
            'support_set'
        ]
        for key in remove_keys:
            output_dict_copy.pop(key, None)
        save_dict(
            dict=output_dict_copy,
            log_dir=inference_cfg['output_root'] / "preds" / f"{pred_hash}.pkl"
        )
    
    return cal_metric_errors_dict


def get_groundtruth_amount(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
):
    num_classes = y_pred.shape[1]
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes) # B x 1 x H x W x C
    label_amounts = y_true_one_hot.sum(dim=(0, 1, 2, 3)) # C
    return {f"num_lab_{i}_pixels": label_amounts[i].item() for i in range(num_classes)}
