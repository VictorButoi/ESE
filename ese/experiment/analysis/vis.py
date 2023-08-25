import torch 
 

def pixelwise_unc_map(subj):
    calibration_image = torch.zeros_like(subj['label']).float()
    foreground_accuracy = (subj['label'] == subj['hard_pred']).float()
    fore_regions = (subj['label'] == 1).bool()
    # Set the regions of the image corresponding to groundtruth label.
    calibration_image[fore_regions] = (foreground_accuracy - subj['soft_pred']).abs()[fore_regions]

    return calibration_image


def ese_unc_map(subj, region_bins, ese_bin_scores):
    pred = subj['soft_pred']
    calibration_image = torch.zeros_like(pred).float()
    bin_width = region_bins[1] - region_bins[0]
    for b_idx, bin in enumerate(region_bins):
        bin_mask = (pred >= bin) & (pred < bin + bin_width)
        calibration_image[bin_mask] = ese_bin_scores[b_idx] 
    return calibration_image
