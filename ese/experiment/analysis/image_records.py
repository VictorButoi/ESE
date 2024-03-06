# torch imports
import torch
from torch.nn import functional as F
# Misc imports
from pydantic import validate_arguments
# local imports
from ..metrics.scoring import pixel_ambiguity, region_ambiguity
from ..metrics.local_ps import bin_stats_init
from ..experiment.utils import reduce_ensemble_preds
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records
):
    num_prob_bins = inference_cfg["local_calibration"]["num_prob_bins"]
    neighborhood_width = inference_cfg["local_calibration"]["neighborhood_width"]

    # (Both individual and reduced)
    if inference_cfg["model"]["ensemble"]:
        # Gather the individual predictions
        if output_dict['y_logits'] is not None:
            output_dict['y_probs'] = torch.softmax(output_dict['y_logits'], dim=1)
            output_dict.pop('y_logits')
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
                "preloaded_obj_dict": bin_stats_init(
                                        y_pred=member_pred,
                                        y_true=output_dict["y_true"],
                                        from_logits=False,
                                        num_prob_bins=num_prob_bins,
                                        neighborhood_width=neighborhood_width
                                    )
            } for member_pred in ensemble_member_preds
        ]
        # Get the reduced predictions
        qual_input_config = {
            'y_pred': reduce_ensemble_preds(
                        output_dict, 
                        inference_cfg=inference_cfg,
                    )['y_probs'],
            'y_true': output_dict['y_true']
        }
        # Get the reduced predictions
        cal_input_config = {
            "preloaded_obj_dict": bin_stats_init(
                                    y_pred=qual_input_config['y_pred'],
                                    y_true=qual_input_config['y_true'],
                                    from_logits=False,
                                    num_prob_bins=num_prob_bins,
                                    neighborhood_width=neighborhood_width
                                ),
            **qual_input_config
        }
    else:
        qual_input_config = {
            "y_pred": output_dict["y_probs"], # either (B, C, H, W) or (B, C, E, H, W), if ensembling
            "y_true": output_dict["y_true"], # (B, C, H, W)
        }
        cal_input_config = {
            **qual_input_config,
            "preloaded_obj_dict": bin_stats_init(
                                    y_pred=qual_input_config['y_pred'],
                                    y_true=qual_input_config['y_true'],
                                    from_logits=False,
                                    num_prob_bins=num_prob_bins,
                                    neighborhood_width=neighborhood_width
                                )
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
    # If we are ensembling, then we also want to calculate the variance of the ensemble predictions.
    if inference_cfg["model"]["ensemble"]:
        # Calculate the variance of the ensemble predictions.
        soft_ensemble_preds = torch.stack(ensemble_member_preds, dim=0) # E x B x C x H x W
        qual_metric_scores_dict["Pixel-Ambiguity"] = pixel_ambiguity(
            ind_preds=soft_ensemble_preds, 
            ens_pred=qual_input_config["y_pred"], 
            from_logits=False
        )
        qual_metric_scores_dict["Region-Ambiguity"] = region_ambiguity(
            ind_preds=soft_ensemble_preds, 
            ens_pred=qual_input_config["y_pred"], 
            from_logits=False
        )
    else:
        qual_metric_scores_dict["Pixel-Ambiguity"] = None
        qual_metric_scores_dict["Region-Ambiguity"] = None

    # If you're showing the predictions, also print the scores.
    if inference_cfg["log"].get("show_examples", False):
        print(f"Ensemble-VAR: {qual_metric_scores_dict['Ensemble-VAR']}")
        print(f"Ensemble-TOP-VAR: {qual_metric_scores_dict['Ensemble-TOP-VAR']}")
        print(f"Avg-PW Soft-Dice: {qual_metric_scores_dict['Avg-PW Soft-Dice']}")
        print(f"Avg-PW Hard-Dice: {qual_metric_scores_dict['Avg-PW Hard-Dice']}")
        print(f"Ambiguity: {qual_metric_scores_dict['Ambiguity']}")

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
                # print("Ignoring Ensemble member preds~!")
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
    
    # Calculate the amount of present ground-truth there is in the image per label.
    if inference_cfg["log"]["track_label_amounts"]:
        pred_cls = "y_probs" if (output_dict["y_probs"] is not None) else "y_logits"
        num_classes = output_dict[pred_cls].shape[1]
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
            metric_score = metric_score_dict[met_name]
            if metric_score is not None:
                metric_score = metric_score.item()
            metrics_record = {
                "image_metric": met_name,
                "metric_score": metric_score
            }
            if inference_cfg["model"]["ensemble"] and met_name not in [
                "Ensemble-VAR", 
                "Ensemble-TOP-VAR", 
                "Avg-PW Soft-Dice",
                "Avg-PW Hard-Dice",
                "Ambiguity"
            ]:
                metrics_record["groupavg_image_metric"] = f"GroupAvg_{met_name}"
                metrics_record["groupavg_metric_score"] = grouped_scores_dict[dict_type][met_name]
            # Add the dataset info to the record
            record = {
                "data_id": output_dict["data_id"],
                "split": output_dict["split"],
                "slice_idx": output_dict["slice_idx"],
                **metrics_record
            }
            if inference_cfg["log"]["track_label_amounts"]:
                record = {**record, **label_amounts_dict}
            # Add the record to the list.
            image_level_records.append(record)
    
    return cal_metric_errors_dict