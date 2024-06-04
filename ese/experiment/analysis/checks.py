import torch


def global_cal_sanity_check(
    data_id,
    slice_idx,
    inference_cfg,
    image_cal_metrics_dict,
    image_tl_pixel_meter_dict,
    image_cw_pixel_meter_dict
):
    # Iterate through all the calibration metrics and check that the pixel level calibration score
    # is the same as the image level calibration score (only true when we are working with a single
    # image.
    for cal_metric_name  in inference_cfg["image_cal_metrics"].keys():
        assert len(cal_metric_name.split("_")) == 2, f"Calibration metric name {cal_metric_name} not formatted correctly."
        metric_base = cal_metric_name.split("_")[-1]
        assert metric_base in inference_cfg["global_cal_metrics"], f"Metric base {metric_base} not found in global calibration metrics."
        global_metric_dict = inference_cfg["global_cal_metrics"][metric_base]
        # Get the calibration error in two views. 
        image_cal_score = image_cal_metrics_dict[cal_metric_name]
        # Choose which pixel meter dict to use.
        if global_metric_dict['cal_type'] == 'classwise':
            # Recalculate the calibration score using the pixel meter dict.
            meter_cal_score = global_metric_dict['_fn'](pixel_meters_dict=image_cw_pixel_meter_dict)
        elif global_metric_dict['cal_type'] == 'toplabel':
            # Recalculate the calibration score using the pixel meter dict.
            meter_cal_score = global_metric_dict['_fn'](pixel_meters_dict=image_tl_pixel_meter_dict)
        else:
            raise ValueError(f"Calibration type {global_metric_dict['cal_type']} not recognized.")
        if torch.abs(image_cal_score - meter_cal_score) >= 1e-3: # Allow for some numerical error.
            raise ValueError(f"WARNING on data id {data_id}, slice {slice_idx}: CALIBRATION METRIC '{cal_metric_name}' DOES NOT MATCH FOR IMAGE AND PIXEL LEVELS."+\
            f" Pixel level calibration score ({meter_cal_score}) does not match image level score ({image_cal_score}).")

