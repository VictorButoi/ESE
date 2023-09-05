import numpy as np
from ese.experiment.metrics import ESE, ReCE


def ECE_map(subj):
    calibration_image = np.zeros_like(subj['label'])
    foreground_accuracy = (subj['label'] == subj['hard_pred']).float()
    fore_regions = (subj['label'] == 1).bool()
    # Set the regions of the image corresponding to groundtruth label.
    calibration_image[fore_regions] = (foreground_accuracy - subj['soft_pred']).abs()[fore_regions]

    return calibration_image


def ESE_map(subj, bins):
    ese_bin_scores, _, _ = ESE(
        bins=bins,
        pred=subj["soft_pred"],
        label=subj["label"],
    ) 

    pred = subj['soft_pred']
    calibration_image = np.zeros_like(pred)

    # Make sure bins are aligned.
    bin_width = bins[1] - bins[0]
    for b_idx, bin in enumerate(bins):
        bin_mask = (pred >= bin) & (pred < (bin + bin_width))
        calibration_image[bin_mask] = ese_bin_scores[b_idx] 

    return calibration_image


def ReCE_map(subj, bins):
    ese_bin_scores, _, _ = ReCE(
        bins=bins,
        pred=subj["soft_pred"],
        label=subj["label"],
    ) 

    pred = subj['soft_pred']
    calibration_image = np.zeros_like(pred)

    # Make sure bins are aligned.
    bin_width = bins[1] - bins[0]
    for b_idx, bin in enumerate(bins):
        bin_mask = (pred >= bin) & (pred < (bin + bin_width))
        calibration_image[bin_mask] = ese_bin_scores[b_idx] 

    return calibration_image