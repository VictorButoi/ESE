# Misc imports
import einops
import numpy as np
import pandas as pd
from tqdm import tqdm

# torch imports
import torch
from torch.utils.data import DataLoader

# local imports
from ese.experiment.metrics import ECE, ESE
from ese.experiment.metrics.utils import reduce_scores
from ese.experiment.experiment.ese_exp import CalibrationExperiment

# ionpy imports
from ionpy.metrics import dice_score, pixel_accuracy
from ionpy.util.torchutils import to_device
from ionpy.util.validation import validate_arguments_init
from ionpy.experiment.util import absolute_import


@validate_arguments_init
def get_dice_breakdown(
        exp: CalibrationExperiment,
        dataset_cfg_list: list,
        num_bins: int = 10
) -> None:

    items = []
    for dataset_cfg in dataset_cfg_list:

        dataset_dict = dataset_cfg['dataset'].to_dict()
        print(f"Processing {dataset_dict['task']} {dataset_dict['split']}")
        
        # Build the dataset and dataloader.
        dataset_cls = absolute_import(dataset_dict.pop("_class"))
        DatasetObj = dataset_cls(**dataset_dict)
        dataloader = DataLoader(DatasetObj, batch_size=1, shuffle=False, drop_last=False)

        with torch.no_grad():
            for subj_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Get your image label pair and define some regions.
                x, y = to_device(batch, exp.device)

                # Reshape so that we will like the shape.
                x_vol = einops.rearrange(x, "b c h w -> (b c) 1 h w")
                y_vol = einops.rearrange(y, "b c h w -> (b c) 1 h w")
                # data_dict[entry_name] = {}

                # Go through each slice and predict the metrics.
                for slice_idx in range(x_vol.shape[0]):

                    # Extract the slices from the volumes.
                    x = x_vol[slice_idx, ...][None]
                    y = y_vol[slice_idx, ...][None]

                    # Get the prediction
                    yhat = exp.model(x) 
                    y_pred = torch.sigmoid(yhat)

                    # Calculate the scores we want to compare against.
                    dice = dice_score(y_pred, y).cpu().numpy().item()
                    acc = pixel_accuracy(y_pred, y).cpu().numpy().item()

                    # Other statistics we might care about.
                    lab_amount = y.sum().cpu().numpy().item()

                    for metric in ["ECE", "ESE"]:
                        for weighting in ["uniform", "proportional"]:
                            if metric == "ECE":
                                ece_bins = np.linspace(0.5, 1, (num_bins//2)+1)[:-1]
                                metric_per_bin, _, bin_counts = ECE(
                                    bins=ece_bins,
                                    pred=y_pred, 
                                    label=y) 
                            else:
                                ese_bins = np.linspace(0, 1, num_bins+1)[:-1]
                                metric_per_bin, _, bin_counts = ESE(
                                    bins=ese_bins,
                                    pred=y_pred, 
                                    label=y)
                            
                            # Calculate the metric score.
                            metric_score = reduce_scores(metric_per_bin, bin_counts, weighting=weighting)

                            # Wrap it in an item
                            items.append({
                                "subj_idx": subj_idx,
                                "slice": slice_idx,
                                "label_amount": lab_amount, 
                                "metric": metric,
                                "metric_weighting": weighting,
                                "metric_score": metric_score,
                                "metric_bins": tuple(metric_per_bin),
                                "bin_counts": tuple(bin_counts),
                                "Acc": acc,
                                "Dice": dice,
                                "task": dataset_dict["task"],
                                "split": dataset_dict["split"]
                            })
        
    # Save the items in a parquet file
    df = pd.DataFrame(items)
    df.to_pickle('/storage/vbutoi/scratch/ESE/records/inference_stats.pkl')