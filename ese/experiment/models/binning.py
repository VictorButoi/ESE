# torch imports
import torch
import torch.nn as nn
# misc imports
import pickle
import matplotlib.pyplot as plt
# local imports
from ..metrics.global_ps import (
    classwise_prob_bin_stats, 
    classwise_neighbor_prob_bin_stats
)
from ..metrics.utils import (
    get_bins, 
    get_conf_region,
    get_bin_per_sample,
    agg_neighbors_preds
)
# Set the print options
torch.set_printoptions(sci_mode=False, precision=3)


class Histogram_Binning(nn.Module):

    def __init__(
        self,
        model_cfg,
        calibration_cfg,
        stats_file: str,
        **kwargs
    ):
        super(Histogram_Binning, self).__init__()
        # Set the parameters as class attributes
        self.num_prob_bins = calibration_cfg['num_prob_bins']
        self.num_classes = calibration_cfg['num_classes']
        self.normalize = model_cfg['normalize']
        self.stats_file = stats_file
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = classwise_prob_bin_stats(
            pixel_meters_dict=pixel_meters_dict[model_cfg['cal_stats_split']], # Use the validation set stats.
            num_prob_bins=self.num_prob_bins, # Use 15 bins
            num_classes=self.num_classes,
            class_wise=True,
            local=False,
            device="cuda"
        )
        self.val_freqs = gbs['bin_freqs'] # C x Bins
        # Get the bins and bin widths
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_prob_bins=self.num_prob_bins, 
            int_start=0.0,
            int_end=1.0
        )

    def forward(self, logits):
        C = self.num_classes
        # Softmax the logits to get probabilities
        prob_tensor = torch.softmax(logits, dim=1) # B x C x H x W
        # Calculate the bin ownership map and transform the probs.
        prob_bin_ownership_map = get_bin_per_sample(
            pred_map=prob_tensor,
            class_wise=True,
            bin_starts=self.conf_bins,
            bin_widths=self.conf_bin_widths
        ) # B x H x W
        for lab_idx in range(C):
            calibrated_lab_prob_map = self.val_freqs[lab_idx][prob_bin_ownership_map[:, lab_idx, ...]] # B x H x W
            # Inserted the calibrated prob map back into the original prob map.
            prob_tensor[:, lab_idx, :, :] = calibrated_lab_prob_map
        # If we are normalizing then we need to make sure the probabilities sum to 1.
        if self.normalize:
            sum_tensor = prob_tensor.sum(dim=1, keepdim=True)
            sum_tensor[sum_tensor == 0] = 1.0
            assert (sum_tensor > 0).all(), "Sum tensor has non-positive values."
            # Return the normalized probabilities.
            return prob_tensor / sum_tensor
        else:
            return prob_tensor 


class Contextual_Histogram_Binning(nn.Module):

    def __init__(
        self,
        model_cfg,
        calibration_cfg,
        **kwargs
    ):
        super(Contextual_Histogram_Binning, self).__init__()
        # Set the parameters as class attributes
        self.num_prob_bins = calibration_cfg['num_prob_bins']
        self.num_classes = calibration_cfg['num_classes']
        self.normalize = model_cfg['normalize']
        # Get the bins and bin widths
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_prob_bins=self.num_prob_bins, 
            int_start=0.0,
            int_end=1.0
        )

    def forward(self, context_images, context_labels, target_logits):
        assert target_logits.shape[0] == 1, "Batch size must be 1 for prediction for now."
        # Predict with the base model.
        C = self.num_classes
        raise ValueError
        # Softmax the logits to get probabilities
        prob_tensor = torch.softmax(logits, dim=1) # B x C x H x W
        # Calculate the bin ownership map and transform the probs.
        prob_bin_ownership_map = get_bin_per_sample(
            pred_map=prob_tensor,
            class_wise=True,
            bin_starts=self.conf_bins,
            bin_widths=self.conf_bin_widths
        ) # B x H x W
        for lab_idx in range(C):
            calibrated_lab_prob_map = self.val_freqs[lab_idx][prob_bin_ownership_map[:, lab_idx, ...]] # B x H x W
            # Inserted the calibrated prob map back into the original prob map.
            prob_tensor[:, lab_idx, :, :] = calibrated_lab_prob_map
        # If we are normalizing then we need to make sure the probabilities sum to 1.
        if self.normalize:
            sum_tensor = prob_tensor.sum(dim=1, keepdim=True)
            sum_tensor[sum_tensor == 0] = 1.0
            assert (sum_tensor > 0).all(), "Sum tensor has non-positive values."
            # Return the normalized probabilities.
            return prob_tensor / sum_tensor
        else:
            return prob_tensor 

    @property
    def device(self):
        return "cpu"