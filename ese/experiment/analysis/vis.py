import torch 
import matplotlib.pyplot as plt
from ese.experiment.metrics import ESE

def pixelwise_unc_map(subj):
    calibration_image = torch.zeros_like(subj['label']).float()
    foreground_accuracy = (subj['label'] == subj['hard_pred']).float()
    fore_regions = (subj['label'] == 1).bool()
    # Set the regions of the image corresponding to groundtruth label.
    calibration_image[fore_regions] = (foreground_accuracy - subj['soft_pred']).abs()[fore_regions]

    return calibration_image


def ese_unc_map(subj, bins):
    ese_bin_scores, _, _ = ESE(
        bins=bins,
        pred=subj["soft_pred"],
        label=subj["label"],
    ) 

    pred = subj['soft_pred']
    calibration_image = torch.zeros_like(pred).float()

    # Make sure bins are aligned.
    bin_width = bins[1] - bins[0]
    for b_idx, bin in enumerate(bins):
        bin_mask = (pred >= bin) & (pred < (bin + bin_width))
        calibration_image[bin_mask] = ese_bin_scores[b_idx] 

    return calibration_image
