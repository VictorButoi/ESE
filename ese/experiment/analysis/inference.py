# Misc imports
import einops
import pathlib
import copy
import pandas as pd
from tqdm import tqdm
from typing import Any, Optional, List
from pydantic import validate_arguments

# torch imports
import torch
from torch.utils.data import DataLoader

# local imports
from ese.experiment.metrics import ACE, ECE, Island_ECE, ReCE
from ese.experiment.experiment.ese_exp import CalibrationExperiment

# ionpy imports
from ionpy.analysis import ResultsLoader
from ionpy.metrics import dice_score, pixel_accuracy
from ionpy.metrics.segmentation import balanced_pixel_accuracy
from ionpy.util import Config
from ionpy.util.config import config_digest
from ionpy.util.torchutils import to_device
from ionpy.experiment.util import absolute_import, generate_tuid


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

    # Import the dataset class
    dataset_cls = absolute_import(dataset_class)

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
            if data_type == "volume":
                volume_forward_loop(batch, batch_idx, cfg, exp, data_records)
            elif data_type == "image":
                image_forward_loop(batch, batch_idx, cfg, exp, data_records)
            else:
                raise ValueError(f"Data type {data_type} not supported.")
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
            slice_idx=slice_idx,
            task=cfg['dataset']['task']
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    batch: Any,
    batch_idx: int,
    cfg: Config,
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
        data_records=data_records
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
            calibration_info = metric_dict[cal_metric](
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
                "cal_score": calibration_info["score"],
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