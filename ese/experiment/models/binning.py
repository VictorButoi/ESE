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
        num_prob_bins: int,
        num_classes: int,
        stats_file: str,
        normalize: bool,
        cal_stats_split: str,
        **kwargs
    ):
        super(Histogram_Binning, self).__init__()
        # Set the parameters as class attributes
        self.num_prob_bins = num_prob_bins
        self.num_classes = num_classes
        self.stats_file = stats_file
        self.normalize = normalize
        self.cal_stats_split = cal_stats_split
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = classwise_prob_bin_stats(
            pixel_meters_dict=pixel_meters_dict[self.cal_stats_split], # Use the validation set stats.
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
            class_wise=False,
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


class Local_Histogram_Binning(nn.Module):

    def __init__(
        self,
        num_prob_bins: int,
        num_classes: int,
        stats_file: str,
        normalize: bool,
        cal_stats_split: str,
        **kwargs
    ):
        super(Histogram_Binning, self).__init__()
        # Set the parameters as class attributes
        self.num_prob_bins = num_prob_bins
        self.num_classes = num_classes
        self.stats_file = stats_file
        self.normalize = normalize
        self.cal_stats_split = cal_stats_split
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = classwise_prob_bin_stats(
            pixel_meters_dict=pixel_meters_dict[self.cal_stats_split], # Use the validation set stats.
            num_prob_bins=self.num_prob_bins, # Use 15 bins
            num_classes=self.num_classes,
            class_wise=True,
            local=True,
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
        # Calculate the continuous neighbor map.
        loc_conf_bin_map = get_bin_per_sample(
            pred_map=agg_neighbors_preds(
                        pred_map=prob_tensor,
                        class_wise=True,
                        neighborhood_width=self.neighborhood_width,
                        discrete=False,
                    ),
            class_wise=True,
            bin_starts=self.conf_bins,
            bin_widths=self.conf_bin_widths,
        )
        for lab_idx in range(C):
            calibrated_lab_prob_map = self.val_freqs[lab_idx][loc_conf_bin_map[:, lab_idx, ...]] # B x H x W
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


class NECTAR_Binning(nn.Module):

    def __init__(
        self,
        num_prob_bins: int,
        num_classes: int,
        neighborhood_width: int,
        stats_file: str,
        normalize: bool,
        cal_stats_split: str,
        **kwargs
    ):
        super(NECTAR_Binning, self).__init__()
        # Set the parameters as class attributes
        self.num_prob_bins = num_prob_bins
        self.num_classes = num_classes
        self.neighborhood_width = neighborhood_width
        self.num_neighbor_classes = neighborhood_width**2
        self.stats_file = stats_file
        self.normalize = normalize
        self.cal_stats_split = cal_stats_split
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = classwise_neighbor_prob_bin_stats(
            pixel_meters_dict=pixel_meters_dict[self.cal_stats_split],
            num_prob_bins=self.num_prob_bins, # Use 15 bins
            num_classes=self.num_classes,
            neighborhood_width=self.neighborhood_width,       
            class_wise=True,
            discrete=True,
            device="cuda"
        )
        self.val_freqs = gbs['bin_freqs'] # C x Neighborhood Classes x Bins
        # Get the bins and bin widths
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_prob_bins=self.num_prob_bins, 
            int_start=0.0,
            int_end=1.0
        )

    def forward(self, logits):
        # Define C as the number of classes.
        C = self.num_classes
        # Softmax the logits to get probabilities
        prob_tensor = torch.softmax(logits, dim=1) # B x C x H x W
        y_hard = torch.argmax(prob_tensor, dim=1).long() # B x H x W
        # Calculate the binary neighbor map.
        disc_neighbor_map = agg_neighbors_preds(
            pred_map=y_hard,
            class_wise=True,
            num_classes=C,
            neighborhood_width=self.neighborhood_width,
            discrete=True,
        ) # B x H x W
        # Calculate the bin ownership map for each pixel probability
        prob_bin_map = get_bin_per_sample(
            pred_map=prob_tensor, 
            class_wise=True,
            bin_starts=self.conf_bins,
            bin_widths=self.conf_bin_widths
        ) # B x H x W
        # Iterate through each label, and replace the probs with the calibrated probs.
        for lab_idx in range(C):
            lab_prob_map = prob_tensor[:, lab_idx, :, :] # B x H x W
            lab_disc_neighbor_map = disc_neighbor_map[:, lab_idx, :, :] # B x H x W
            lab_prob_bin_map = prob_bin_map[:, lab_idx, :, :] # B x H x W
            # We are building the calibrated prob map. 
            calibrated_lab_prob_map = torch.zeros_like(lab_prob_map)
            # Construct the prob_maps
            for nn_idx in range(self.num_neighbor_classes):
                neighbor_cls_mask = (lab_disc_neighbor_map==nn_idx)
                # Replace the soft predictions with the old frequencies.
                calibrated_lab_prob_map[neighbor_cls_mask] =\
                    self.val_freqs[lab_idx][nn_idx][lab_prob_bin_map][neighbor_cls_mask].float()
            # Inserted the calibrated prob map back into the original prob map.
            prob_tensor[:, lab_idx, :, :] = calibrated_lab_prob_map

        # If we are normalizing then we need to make sure the probabilities sum to 1.
        if self.normalize:
            sum_tensor = prob_tensor.sum(dim=1, keepdim=True)
            sum_tensor[sum_tensor == 0] = 1
            assert (sum_tensor > 0).all(), "Sum tensor has non-positive values."
            # Return the normalized probabilities.
            return prob_tensor / sum_tensor
        else:
            return prob_tensor 

    @property
    def device(self):
        return "cpu"


class Soft_NECTAR_Binning(nn.Module):

    def __init__(
        self,
        num_prob_bins: int,
        num_classes: int,
        neighborhood_width: int,
        stats_file: str,
        normalize: bool,
        cal_stats_split: str,
        **kwargs
    ):
        super(Soft_NECTAR_Binning, self).__init__()
        # Set the parameters as class attributes
        self.num_prob_bins = num_prob_bins
        self.num_classes = num_classes
        self.neighborhood_width = neighborhood_width
        self.num_neighbor_classes = neighborhood_width**2
        self.stats_file = stats_file
        self.normalize = normalize
        self.cal_stats_split = cal_stats_split
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = classwise_neighbor_prob_bin_stats(
            pixel_meters_dict=pixel_meters_dict[self.cal_stats_split],
            num_prob_bins=self.num_prob_bins, # Use 15 bins
            num_classes=self.num_classes,
            neighborhood_width=self.neighborhood_width,       
            class_wise=True,
            discrete=False,
            device="cuda"
        )
        self.val_freqs = gbs['bin_freqs'] # C x Neighborhood Classes x Bins
        # raise ValueError
        # Get the bins and bin widths
        self.neighbor_bins, self.neighbor_bin_widths = get_bins(
            num_prob_bins=self.num_neighbor_classes, 
            int_start=0.0,
            int_end=1.0
        )
        # Get the bins and bin widths
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_prob_bins=self.num_prob_bins, 
            int_start=0.0,
            int_end=1.0
        )

    def forward(self, logits):
        # Softmax the logits to get probabilities
        prob_tensor = torch.softmax(logits, dim=1) # B x C x H x W
        # Calculate the continuous neighbor map.
        loc_conf_bin_map = get_bin_per_sample(
            pred_map=agg_neighbors_preds(
                        pred_map=prob_tensor,
                        class_wise=True,
                        neighborhood_width=self.neighborhood_width,
                        discrete=False,
                    ),
            class_wise=True,
            bin_starts=self.neighbor_bins,
            bin_widths=self.neighbor_bin_widths,
        )
        # Calculate the bin ownership map for each pixel probability
        prob_bin_map = get_bin_per_sample(
            pred_map=prob_tensor, 
            class_wise=True,
            bin_starts=self.conf_bins,
            bin_widths=self.conf_bin_widths
        ) # B x H x W
        # Iterate through each label, and replace the probs with the calibrated probs.
        for lab_idx in range(self.num_classes):
            lab_prob_map = prob_tensor[:, lab_idx, :, :] # B x H x W
            lab_loc_conf_bin_map = loc_conf_bin_map[:, lab_idx, :, :] # B x H x W
            lab_prob_bin_map = prob_bin_map[:, lab_idx, :, :] # B x H x W
            # We are building the calibrated prob map. 
            calibrated_lab_prob_map = torch.zeros_like(lab_prob_map)
            for nn_bin_idx in range(self.num_neighbor_classes):
                # Get the region of image corresponding to the confidence
                nn_conf_region = get_conf_region(
                    conditional_region_dict={
                        "bin_idx": (nn_bin_idx, lab_loc_conf_bin_map)
                    }
                )
                calibrated_lab_prob_map[nn_conf_region] =\
                    self.val_freqs[lab_idx][nn_bin_idx][lab_prob_bin_map][nn_conf_region].float()
            # Inserted the calibrated prob map back into the original prob map.
            prob_tensor[:, lab_idx, :, :] = calibrated_lab_prob_map
        # If we are normalizing then we need to make sure the probabilities sum to 1.
        if self.normalize:
            sum_tensor = prob_tensor.sum(dim=1, keepdim=True)
            sum_tensor[sum_tensor == 0] = 1
            assert (sum_tensor > 0).all(), "Sum tensor has non-positive values."
            # Return the normalized probabilities.
            return prob_tensor / sum_tensor
        else:
            return prob_tensor 

    @property
    def device(self):
        return "cpu"