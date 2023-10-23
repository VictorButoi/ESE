# Misc imports
import copy
import einops
import pathlib
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from pydantic import validate_arguments
from typing import Any, Dict, Optional, List, Tuple
# torch imports
import torch
from torch.utils.data import DataLoader
# ionpy imports
from ionpy.util import Config, MeterDict
from ionpy.analysis import ResultsLoader
from ionpy.util.config import config_digest
from ionpy.util.torchutils import to_device
from ionpy.metrics import dice_score, pixel_accuracy
from ionpy.metrics.segmentation import balanced_pixel_accuracy
from ionpy.experiment.util import absolute_import, generate_tuid
# local imports
from .utils import count_matching_neighbors
from ..metrics.utils.utils import get_bins, find_bins
from ..experiment.ese_exp import CalibrationExperiment


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def compute_pixel_meter_dict(
    data_points: List[dict],
    num_bins: int,
    conf_interval: List[float]
    ) -> dict[MeterDict]:
    # Get the different confidence bins.
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Go through each datapoint and register information about the pixels.
    meter_dict ={}
    # for label in unique_labels:
    #     meter_dict[label] = MeterDict()
    # Build the dict.
    
    return meter_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_pixelinfo_df(
    data_points: List[dict],
    num_bins: int,
    conf_interval: List[float]
    ) -> pd.DataFrame:
    # Get the different confidence bins.
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Go through each datapoint and register information about the pixels.
    perppixel_records = []
    for dp in tqdm(data_points, total=len(data_points)):
        # Get the interesting info.
        accuracy_map = dp['perpix_accuracies']
        bin_info_map = find_bins(
            confidences=dp['conf_map'], 
            bin_starts=conf_bins,
            bin_widths=conf_bin_widths
            )
        # iterate through (x,y) positions from 0 - 255, 0 - 255
        for (ix, iy) in np.ndindex(dp['pred_map'].shape):
            # Record all important info.
            if bin_info_map[ix, iy] > 0: # Weird bug where we fall in no bin.
                bin_n = int(bin_info_map[ix, iy].item()) - 1
                bin_ends = [np.round(conf_bins[bin_n].item(), 2), np.round(conf_bins[bin_n].item() + conf_bin_widths[bin_n].item(), 2)]
                for measure in ["conf", "accuracy"]:
                    val = dp['conf_map'][ix, iy] if measure == "conf" else accuracy_map[ix, iy]
                    record = {
                        "subject_id": dp['subject_id'],
                        "x": ix,
                        "y": iy,
                        "bin_num": bin_n + 1,
                        "bin": f"{bin_n + 1}, {bin_ends}",
                        "pred_label": dp['pred_map'][ix, iy],
                        "num_neighbors": dp['matching_neighbors'][ix, iy],
                        "pred_label,num_neighbors": f"{dp['pred_map'][ix, iy]},{dp['matching_neighbors'][ix, iy]}",
                        "measure": measure,
                        "value": val,
                    }
                    perppixel_records.append(record)
    return pd.DataFrame(perppixel_records) 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_dataset_perf(
    exp, 
    split="val",
    threshold=0.5,
    rearrange_channel_dim=False
    ) -> Tuple[List[dict], List[int]]:
    # Choose the dataloader from the experiment.
    dataloader = exp.val_dl if split=="val" else exp.train_dl
    items = []
    unique_labels = set([])
    with torch.no_grad():
        for subj_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Get your image label pair and define some regions.
            x, y = to_device(batch, exp.device)

            # Some datasets where we consider multi-slice we want to batch different slices.
            if rearrange_channel_dim:
                x = einops.rearrange(x, "b c h w -> (b c) 1 h w")
                y = einops.rearrange(y, "b c h w -> (b c) 1 h w")

            # Get the prediction
            y_logits = exp.model(x)  

            # Extract predictions
            n_channels = y_logits.shape[1]
            if n_channels > 1:
                y_hat = torch.softmax(y_logits, dim=1) 
                y_pred = torch.argmax(y_hat, dim=1)
            else:
                y_hat = torch.sigmoid(y_logits)
                y_pred = (y_hat > threshold).float()

            # Get the max conf map (flip background for binary case).
            if n_channels > 1:
                probs = torch.max(y_hat, dim=1).values
            else:
                probs = y_hat.clone()
                probs[probs < 0.5] = 1 - probs[probs < 0.5]
            
            # Squeeze our tensors
            image = x.squeeze().cpu().numpy()
            label_map = y.squeeze().cpu().numpy()
            conf_map = probs.squeeze().cpu().numpy()
            pred_map = y_pred.squeeze().cpu().numpy() 

            # Get the unique labels and merge using set union.
            new_unique_labels = set(np.unique(label_map))
            unique_labels = unique_labels.union(new_unique_labels)

            # Get some performance metrics
            accuracy_map = (pred_map == label_map).astype(np.float32)
            matching_neighbors = count_matching_neighbors(pred_map)

            # Wrap it in an item
            items.append({
                "subject_id": subj_idx,
                "image": image,
                "label_map": label_map,
                "conf_map": conf_map,
                "pred_map": pred_map,
                "matching_neighbors": matching_neighbors,
                "perpix_accuracies": accuracy_map,
            })
    # Return the subject info and labels.
    return items, list(unique_labels)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_cal_stats(
    cfg: Config
    ) -> None:
    # Get the config dictionary
    cfg_dict = copy.deepcopy(cfg.to_dict())
    # Results loader object does everything
    root = pathlib.Path(cfg['log']['root'])
    exp_name = cfg['model']['exp_name']
    exp_path = root / exp_name
    # Get the configs of the experiment
    rs = ResultsLoader()
    dfc = rs.load_configs(
        exp_path,
        properties=False,
    )
    # Get the best experiment at the checkpoint
    # corresponding to the best metric.
    exp = rs.get_experiment(
        df=rs.load_metrics(dfc),
        exp_class=CalibrationExperiment,
        metric=cfg['model']['metric'],
        checkpoint=cfg['model']['checkpoint'],
        device="cuda"
    )
    # Get information about the dataset
    dataset_dict = cfg['dataset'].to_dict()
    dataset_class = dataset_dict.pop("_class")
    dataset_name = dataset_class.split(".")[-1]
    data_type = dataset_dict.pop("data_type")
    assert data_type in ["volume", "image"], f"Data type {data_type} not supported."

    # Import the dataset class
    dataset_cls = absolute_import(dataset_class)

    # Import the caibration metrics.
    metric_cfg = cfg['cal_metrics'].to_dict()
    for cal_metric in metric_cfg.keys():
        metric_cfg[cal_metric]['func'] = absolute_import(metric_cfg[cal_metric]['func'])

    if 'task' in dataset_dict.keys():
        print(f"Processing {dataset_name}, task: {dataset_dict['task']}, split: {dataset_dict['split']}")
    else:
        print(f"Processing {dataset_name}, split: {dataset_dict['split']}")
    
    # Build the dataset and dataloader.
    DatasetObj = dataset_cls(**dataset_dict)
    dataloader = DataLoader(
        DatasetObj, 
        batch_size=1, 
        num_workers=cfg['model']['num_workers'], 
        shuffle=False
    )

    # Prepare the output dir for saving the results
    create_time, nonce = generate_tuid()
    digest = config_digest(cfg_dict)
    uuid = f"{create_time}-{nonce}-{digest}.pkl"
    exp_log_dir = root / "records" / exp_name
    cfg_log_dir = exp_log_dir / uuid
    if not exp_log_dir.exists():
        exp_log_dir.mkdir(parents=True)

    # Keep track of records
    data_records = []

    def save_records(records):
        # Save the items in a pickle file.  
        df = pd.DataFrame(records)
        # Save or overwrite the file.
        df.to_pickle(cfg_log_dir)

    # Loop through the data, gather your stats!
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), desc="Data Loop", total=len(dataloader)):
            forward_loop_func = volume_forward_loop if data_type == "volume" else image_forward_loop
            # Run the forward loop
            forward_loop_func(
                batch=batch, 
                batch_idx=batch_idx, 
                cfg=cfg, 
                metric_cfg=metric_cfg,
                exp=exp, 
                data_records=data_records
            )
            # Save the records every so often, to get intermediate results.
            if batch_idx % cfg['log']['log_interval'] == 0:
                save_records(data_records)

    # Save the records at the end too
    save_records(data_records)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    batch: Any,
    batch_idx: int,
    cfg: Config,
    metric_cfg: dict,
    exp: CalibrationExperiment,
    data_records: list
):
    # Get your image label pair and define some regions.
    batch_x, batch_y = to_device(batch, exp.device)

    # Reshape so that we will like the shape.
    x_vol = einops.rearrange(batch_x, "b c h w -> (b c) 1 h w")
    y_vol = einops.rearrange(batch_y, "b c h w -> (b c) 1 h w")

    # Go through each slice and predict the metrics.
    for slice_idx in tqdm(range(x_vol.shape[0]), desc="Slice loop", position=1, leave=False):
        
        # Extract the slices from the volumes.
        image = x_vol[slice_idx, ...][None]
        label = y_vol[slice_idx, ...][None]

        # Get the prediction
        pred_map, conf_map = exp.predict(image, include_probs=True)

        get_calibration_item_info(
            cfg=cfg,
            conf_map=conf_map,
            pred_map=pred_map,
            label=label,
            data_idx=batch_idx,
            split=cfg['dataset']['split'],
            data_records=data_records,
            metric_cfg=metric_cfg,
            slice_idx=slice_idx,
            task=cfg['dataset']['task']
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    batch: Any,
    batch_idx: int,
    cfg: Config,
    metric_cfg: dict,
    exp: CalibrationExperiment,
    data_records: list
):
    # Get your image label pair and define some regions.
    image, label = to_device(batch, exp.device)

    # Get the prediction
    pred_map, conf_map = exp.predict(image, include_probs=True)

    get_calibration_item_info(
        cfg=cfg,
        conf_map=conf_map,
        pred_map=pred_map,
        label=label,
        data_idx=batch_idx,
        split=cfg['dataset']['split'],
        data_records=data_records,
        metric_cfg=metric_cfg
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    cfg: Config,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label: torch.Tensor,
    data_idx: int,
    split: str,
    data_records: list,
    metric_cfg: dict,
    slice_idx: Optional[int] = None,
    task: Optional[str] = None
    ):

    # Calculate the scores we want to compare against.
    gt_lab_amount = label.sum().item()
    amount_pred_lab = pred_map.sum().item()
    
    # Get some metrics of these predictions
    dice = dice_score(conf_map, label).item()
    acc = pixel_accuracy(conf_map, label).item()
    balanced_acc = balanced_pixel_accuracy(conf_map, label).item()
    
    # Squeeze the tensors
    conf_map = conf_map.squeeze()
    pred_map = pred_map.squeeze()
    label = label.squeeze()

    for cal_metric in cfg["calibration"]["metrics"]:
        for bin_weighting in cfg["calibration"]["bin_weightings"]:
            # Get the calibration metric
            calibration_info = metric_cfg[cal_metric]['func'](
                num_bins=cfg["calibration"]["num_bins"],
                conf_map=conf_map,
                pred_map=pred_map,
                label=label,
                class_type=cfg["calibration"]["class_type"],
                weighting=bin_weighting,
                include_background=cfg["calibration"]["include_background"]
            ) 
            # Wrap it in an item
            record = {
                "accuracy": acc,
                "bin_weighting": bin_weighting,
                "cal_metric": cal_metric,
                "cal_score": calibration_info["cal_score"],
                "class_type": cfg["calibration"]["class_type"],
                "data_idx": data_idx,
                "dataset": cfg["dataset"]["_class"],
                "dice": dice,
                "gt_lab_amount": gt_lab_amount, 
                "lab_w_accuracy": balanced_acc,
                "num_bins": cfg["calibration"]["num_bins"],
                "pred_lab_amount": amount_pred_lab,
                "slice_idx": slice_idx,
                "split": split,
                "task": task,
            }
            data_records.append(record)