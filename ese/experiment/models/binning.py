# torch imports
import torch
import torch.nn as nn
# misc imports
import pickle
from dataclasses import dataclass
# ionpy imports
from ionpy.util.validation import validate_arguments_init
# local imports
from ..metrics.global_ps import global_binwise_stats
from ..metrics.utils import (
    get_bins, 
    find_bins, 
    get_conf_region,
    agg_neighbors_preds
)
# Set the print options
torch.set_printoptions(sci_mode=False, precision=3)


@validate_arguments_init
@dataclass
class Histogram_Binning(nn.Module):

    num_bins: int
    num_classes: int
    stats_file: str
    normalize: bool
    cal_stats_split: str

    def __post_init__(self):
        super(Histogram_Binning, self).__init__()
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = global_binwise_stats(
            pixel_meters_dict=pixel_meters_dict[self.cal_stats_split], # Use the validation set stats.
            num_bins=self.num_bins, # Use 15 bins
            num_classes=self.num_classes,
            class_conditioned=True,
            neighborhood_conditioned=False,
            class_wise=True,
            device="cuda"
        )
        self.val_freqs = gbs['bin_freqs'] # C x Bins
        # Get the bins and bin widths
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_bins=self.num_bins, 
            start=0.0,
            end=1.0
        )

    def forward(self, logits):
        C = self.num_classes
        # Softmax the logits to get probabilities
        prob_tensor = torch.softmax(logits, dim=1) # B x C x H x W
        for lab_idx in range(C):
            prob_map = prob_tensor[:, lab_idx, :, :] # B x H x W
            # Calculate the bin ownership map and transform the probs.
            prob_bin_ownership_map = find_bins(
                confidences=prob_map, 
                bin_starts=self.conf_bins,
                bin_widths=self.conf_bin_widths
            ) # B x H x W
            calibrated_prob_map = self.val_freqs[lab_idx][prob_bin_ownership_map] # B x H x W
            # Inserted the calibrated prob map back into the original prob map.
            prob_tensor[:, lab_idx, :, :] = calibrated_prob_map
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


@validate_arguments_init
@dataclass
class NECTAR_Binning(nn.Module):

    num_bins: int
    num_classes: int
    neighborhood_width: int
    discretized_neighbors: bool
    stats_file: str
    normalize: bool 
    cal_stats_split: str

    def __post_init__(self):
        super(NECTAR_Binning, self).__init__()
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = global_binwise_stats(
            pixel_meters_dict=pixel_meters_dict[self.cal_stats_split],
            num_bins=self.num_bins, # Use 15 bins
            num_classes=self.num_classes,
            neighborhood_width=self.neighborhood_width,       
            class_conditioned=True,
            neighborhood_conditioned=True,
            class_wise=True,
            device="cuda"
        )
        self.val_freqs = gbs['bin_freqs'] # C x Neighborhood Classes x Bins
        # Get the bins and bin widths
        if self.discretized_neighbors:
            self.conf_bins, self.conf_bin_widths = get_bins(
                num_bins=self.num_bins, 
                start=0.0,
                end=1.0
            )
        else:
            self.neighbor_bins, self.neighbor_bin_widths = get_bins(
                num_bins=(self.neighborhood_width**2), 
                start=0.0,
                end=1.0
            )

    def forward(self, logits):
        # Define C as the number of classes.
        C = self.num_classes

        # Softmax the logits to get probabilities
        prob_tensor = torch.softmax(logits, dim=1) # B x C x H x W
        y_hard = torch.argmax(prob_tensor, dim=1) # B x H x W

        # Iterate through each label, and replace the probs with the calibrated probs.
        for lab_idx in range(C):
            lab_prob_map = prob_tensor[:, lab_idx, :, :] # B x H x W
            # Calculate the bin ownership map for each pixel probability
            lab_prob_bin_map = find_bins(
                confidences=lab_prob_map, 
                bin_starts=self.conf_bins,
                bin_widths=self.conf_bin_widths
            ) # B x H x W

            # We are building the calibrated prob map. 
            calibrated_lab_prob_map = torch.zeros_like(lab_prob_map)

            # Calculate how many neighbors of each pixel have this label.
            if self.discretized_neighbors:
                disc_neighbor_agg_map = agg_neighbors_preds(
                    lab_map=(y_hard==lab_idx).long(),
                    neighborhood_width=self.neighborhood_width,
                    discrete=True,
                    binary=True
                ) # B x H x W
                # Construct the prob_maps
                for nn_idx in range(self.neighborhood_width**2):
                    neighbor_cls_mask = (disc_neighbor_agg_map==nn_idx)
                    # Replace the soft predictions with the old frequencies.
                    calibrated_lab_prob_map[neighbor_cls_mask] =\
                        self.val_freqs[lab_idx][nn_idx][lab_prob_bin_map][neighbor_cls_mask].float()
            else:
                cont_neighbor_agg_map = agg_neighbors_preds(
                    lab_map=lab_prob_map,
                    neighborhood_width=self.neighborhood_width,
                    discrete=False,
                    binary=True
                )
                neighbor_bin_map = find_bins(
                    confidences=cont_neighbor_agg_map,
                    bin_starts=self.neighbor_bins,
                    bin_widths=self.neighbor_bin_widths
                )
                # Replace the soft predictions with the old freqencies.
                for nn_bin_idx in range(self.neighborhood_width**2):
                    # Get the region of image corresponding to the confidence
                    nn_conf_region = get_conf_region(
                        bin_idx=nn_bin_idx, 
                        bin_ownership_map=neighbor_bin_map,
                    )
                    calibrated_lab_prob_map[nn_conf_region] =\
                        self.val_freqs[lab_idx][nn_bin_idx][lab_prob_bin_map][nn_conf_region].float()
            # Inserted the calibrated prob map back into the original prob map.
            prob_tensor[:, lab_idx, :, :] = calibrated_lab_prob_map

        # If we are normalizing then we need to make sure the probabilities sum to 1.
        if self.normalize:
            sum_tensor = prob_tensor.sum(dim=1, keepdim=True)
            sum_tensor[sum_tensor == 0] = self.smoothing
            assert (sum_tensor > 0).all(), "Sum tensor has non-positive values."
            # Return the normalized probabilities.
            return prob_tensor / sum_tensor
        else:
            return prob_tensor 

    @property
    def device(self):
        return "cpu"