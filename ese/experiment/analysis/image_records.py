# Misc imports
import numpy as np
from pydantic import validate_arguments
# torch imports
from torch.nn import functional as F
# local imports
from .analysis_utils.inference_utils import (
    get_image_aux_info, 
    reduce_ensemble_preds
)
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records
):
    # Define the cal config.
    qual_input_config = {
        "y_pred": output_dict["y_pred"], # either (B, C, H, W) or (B, C, E, H, W), if ensembling
        "y_true": output_dict["y_true"], # (B, C, H, W)
    }
    # Define the cal config.
    cal_input_config = qual_input_config.copy() 
    # If not ensembling, we can cache information.
    if not inference_cfg["model"]["ensemble"]:
        cal_input_config["stats_info_dict"] = get_image_aux_info(
            y_pred=output_dict["y_pred"],
            y_hard=output_dict["y_hard"],
            y_true=output_dict["y_true"],
            cal_cfg=inference_cfg["calibration"]
        )
    # If we are ensembling, then we can precalulate the ensemble predictions.
    # (Both individual and reduced)
    if inference_cfg["model"]["ensemble"]:
        # Get the reduced predictions
        ensemble_input_config = {
            'y_pred': reduce_ensemble_preds(
                output_dict, 
                inference_cfg=inference_cfg)['y_pred'],
            'y_true': output_dict['y_true']
        }
        # Gather the individual predictions
        ensemble_member_preds = [
            output_dict["y_pred"][:, :, ens_mem_idx, ...]\
            for ens_mem_idx in range(output_dict["y_pred"].shape[2])
            ]
        # Construct the input cfgs used for calulating metrics.
        ensemble_member_input_cfgs = [
            {
                "y_pred": member_pred, 
                "y_true": output_dict["y_true"],
                "from_logits": True # IMPORTANT, we haven't softmaxed yet.
            } for member_pred in ensemble_member_preds
        ]
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
                for ens_mem_input_cfg in ensemble_member_input_cfgs:
                    member_qual_score = qual_metric_dict['_fn'](**ens_mem_input_cfg).item()
                    individual_qual_scores.append(member_qual_score)
                # Now place it in the dictionary.
                grouped_scores_dict['quality'][qual_metric_name] = np.mean(individual_qual_scores)
            else:
                grouped_scores_dict['quality'][qual_metric_name] = None
            # Now get the ensemble quality score.
            qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**ensemble_input_config).item() 
        else:
            # Get the calibration error. 
            if qual_metric_dict['_type'] == 'calibration':
                # Higher is better for scores.
                qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**cal_input_config).item() 
            else:
                qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**qual_input_config).item()
            # If you're showing the predictions, also print the scores.
            if inference_cfg["log"]["show_examples"]:
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
                for ens_mem_input_cfg in ensemble_member_input_cfgs:
                    member_cal_score = cal_metric_dict['_fn'](**ens_mem_input_cfg).item()
                    individual_cal_scores.append(member_cal_score)
                # Now place it in the dictionary.
                grouped_scores_dict['calibration'][cal_metric_name] = np.mean(individual_cal_scores)
            else:
                grouped_scores_dict['calibration'][cal_metric_name] = None
            # Now get the ensemble calibration error.
            cal_metric_errors_dict[cal_metric_name] = cal_metric_dict['_fn'](**ensemble_input_config).item() 
        else:
            # Get the calibration error. 
            cal_metric_errors_dict[cal_metric_name] = cal_metric_dict['_fn'](**cal_input_config).item() 
    
    assert not (len(qual_metric_scores_dict) == 0 and len(cal_metric_errors_dict) == 0), \
        "No metrics were specified in the config file."
    
    # Calculate the amount of present ground-truth there is in the image per label.
    if inference_cfg["log"]["track_label_amounts"]:
        num_classes = output_dict["y_pred"].shape[1]
        y_true_one_hot = F.one_hot(output_dict["y_true"], num_classes=num_classes) # B x 1 x H x W x C
        label_amounts = y_true_one_hot.sum(dim=(0, 1, 2, 3)) # C
        label_amounts_dict = {f"num_lab_{i}_pixels": label_amounts[i].item() for i in range(num_classes)}
    
    # Add our scores to the image level records.
    metrics_collection ={
        "quality": qual_metric_scores_dict,
        "calibration": cal_metric_errors_dict
    }

    for dict_type, metric_score_dict in metrics_collection.items():
        for met_name in list(metric_score_dict.keys()):
            metrics_record = {
                "image_metric": met_name,
                "metric_score": metric_score_dict[met_name],
            }
            if inference_cfg["model"]["ensemble"]:
                metrics_record["groupavg_image_metric"] = f"GroupAvg_{met_name}"
                metrics_record["groupavg_metric_score"] = grouped_scores_dict[dict_type][met_name]
            # Add the dataset info to the record
            record = {
                "data_id": output_dict["data_id"],
                "split": output_dict["split"],
                "slice_idx": output_dict["slice_idx"],
                **metrics_record, 
                **inference_cfg["calibration"]
            }
            if inference_cfg["log"]["track_label_amounts"]:
                record = {**record, **label_amounts_dict}
            # Add the record to the list.
            image_level_records.append(record)
    
    return cal_metric_errors_dict