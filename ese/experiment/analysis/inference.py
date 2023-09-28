# Misc imports
import einops
import numpy as np
import pandas as pd
from tqdm import tqdm
from pydantic import validate_arguments

# torch imports
import torch
from torch.utils.data import DataLoader

# local imports
from ese.experiment.metrics import ACE, ECE, ReCE
from ese.experiment.metrics.utils import reduce_scores
from ese.experiment.experiment.ese_exp import CalibrationExperiment

# ionpy imports
from ionpy.analysis import ResultsLoader
from ionpy.metrics import dice_score, pixel_accuracy
from ionpy.metrics.segmentation import balanced_pixel_accuracy
from ionpy.util import Config
from ionpy.util.torchutils import to_device
from ionpy.experiment.util import absolute_import

# Globally used for which metrics to plot for.
metric_dict = {
        "ACE": ACE,
        "ECE": ECE,
        "ReCE": ReCE
    }

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_dice_breakdown(
    cfg: Config,
    root: str = "/storage/vbutoi/scratch/ESE"
    ) -> None:

    # Results loader object does everything
    rs = ResultsLoader()
    path = f"{root}/{cfg['model']['exp_name']}"

    dfc = rs.load_configs(
        path,
        properties=False,
    )

    df = rs.load_metrics(dfc)

    exp = rs.get_experiment(
        df=df,
        exp_class=CalibrationExperiment,
        metric=cfg['model']['metric'],
        checkpoint=cfg['model']['checkpoint'],
        device="cuda"
    )

    dataset_dict = cfg['dataset'].to_dict()
    print(f"Processing {dataset_dict['task']} {dataset_dict['split']}")
    
    # Build the dataset and dataloader.
    dataset_cls = absolute_import(dataset_dict.pop("_class"))
    DatasetObj = dataset_cls(**dataset_dict)
    dataloader = DataLoader(DatasetObj, batch_size=1, shuffle=False, drop_last=False)

    # Keep track of records
    items = []

    with torch.no_grad():
        for subj_idx, batch in tqdm(enumerate(dataloader), desc="Subject Loop", total=len(dataloader)):
            
            # Get your image label pair and define some regions.
            x, y = to_device(batch, exp.device)

            # Reshape so that we will like the shape.
            x_vol = einops.rearrange(x, "b c h w -> (b c) 1 h w")
            y_vol = einops.rearrange(y, "b c h w -> (b c) 1 h w")

            # Go through each slice and predict the metrics.
            for slice_idx in tqdm(range(x_vol.shape[0]), desc="Slice loop", position=1, leave=False):
                
                # Extract the slices from the volumes.
                x = x_vol[slice_idx, ...][None]
                y = y_vol[slice_idx, ...][None]

                # Get the prediction
                yhat = exp.model(x) 
                conf_map = torch.sigmoid(yhat)
                pred_map = (conf_map >= 0.5).float()

                # Calculate the scores we want to compare against.
                lab_amount = y.sum().cpu().numpy().item()
                lab_predicted = pred_map.sum().cpu().numpy().item()
                
                # Get some metrics of these predictions
                dice = dice_score(conf_map, y).cpu().numpy().item()
                acc = pixel_accuracy(conf_map, y).cpu().numpy().item()
                balanced_acc = balanced_pixel_accuracy(conf_map, y).cpu().numpy().item()
                
                # Squeeze the tensors
                conf_map = conf_map.squeeze()
                pred_map = pred_map.squeeze()
                y = y.squeeze()

                for cal_metric in cfg["calibration"]["metrics"]:
                    for bin_weighting in cfg["calibration"]["bin_weightings"]:
                        # Get the calibration metric
                        calibration_info = metric_dict[cal_metric](
                            num_bins=cfg["calibration"]["num_bins"],
                            conf_map=conf_map,
                            pred_map=pred_map,
                            label=y,
                            class_type=cfg["calibration"]["class_type"],
                            weighting=bin_weighting,
                            include_background=cfg["calibration"]["include_background"]
                        ) 
                        # Wrap it in an item
                        items.append({
                            "accuracy": acc,
                            "bin_counts": tuple(calibration_info["bin_amounts"]),
                            "bin_weighting": bin_weighting,
                            "cal_metric": cal_metric,
                            "cal_score": calibration_info["score"],
                            "cal_per_bin": tuple(calibration_info["bin_scores"]),
                            "class_type": cfg["calibration"]["class_type"],
                            "dice": dice,
                            "gt_lab_amount": lab_amount, 
                            "pred_lab_amount": lab_predicted,
                            "slice": slice_idx,
                            "split": dataset_dict["split"],
                            "subj_idx": subj_idx,
                            "task": dataset_dict["task"],
                            "lab_w_accuracy": balanced_acc,
                        })
        
    # Save the items in a parquet file
    df = pd.DataFrame(items)
    df.to_pickle('/storage/vbutoi/scratch/ESE/records/inference_stats.pkl')
