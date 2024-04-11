# torch imports
import torch
import torch.nn as nn
# misc imports
import pickle
import matplotlib.pyplot as plt
# local imports
from ..analysis.pixel_records import update_prob_pixel_meters
from ..metrics.global_ps import (
    prob_bin_stats,
    classwise_prob_bin_stats, 
)
from ..metrics.utils import (
    get_bins, 
    get_bin_per_sample,
    agg_neighbors_preds
)
# Set the print options
torch.set_printoptions(sci_mode=False, precision=3)


class Histogram_Binning(nn.Module):

    def __init__(
        self,
        cal_model_cfg,
        calibration_cfg,
        stats_file: str,
        **kwargs
    ):
        super(Histogram_Binning, self).__init__()
        # Set the parameters as class attributes
        self.num_prob_bins = calibration_cfg['num_prob_bins']
        self.num_classes = calibration_cfg['num_classes']
        self.normalize = cal_model_cfg['normalize']
        self.stats_file = stats_file
        # Load the data from the .pkl file
        with open(self.stats_file, "rb") as f:
            pixel_meters_dict = pickle.load(f)
        # Get the statistics either from images or pixel meter dict.
        gbs = classwise_prob_bin_stats(
            pixel_meters_dict=pixel_meters_dict[cal_model_cfg['cal_stats_split']], # Use the validation set stats.
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
        base_model,
        calibration_cfg,
        **kwargs
    ):
        super(Contextual_Histogram_Binning, self).__init__()
        # Set the parameters as class attributes
        self.base_model = base_model
        self.calibration_cfg = calibration_cfg
        # Get the bins and bin widths
        self.conf_bins, self.conf_bin_widths = get_bins(
            num_prob_bins=self.calibration_cfg['num_prob_bins'], 
            int_start=0.0,
            int_end=1.0
        )
    
    def forward(self, context_images, context_labels, target_logits):
        assert target_logits.shape[0] == 1, "Batch size must be 1 for prediction for now."
        support_pred_meter_dict = {} 
        for i in range(context_images.shape[1]):
            # Get the new support by removing the ith image from the context set.
            new_target_image = context_images[:, i, ...]
            new_context_images = torch.cat([context_images[:, :i, ...], context_images[:, i+1:, ...]], dim=1)
            new_context_labels = torch.cat([context_labels[:, :i, ...], context_labels[:, i+1:, ...]], dim=1)
            # Predict with the base model.
            support_target_logits = self.base_model(
                context_images=new_context_images, 
                context_labels=new_context_labels, 
                target_image=new_target_image
            )
            y_probs = torch.sigmoid(support_target_logits)
            # Optionally smooth the probs.
            if self.calibration_cfg['conf_pool_width'] > 1:
                y_probs = agg_neighbors_preds(
                    pred_map=y_probs, # B x H x W
                    neighborhood_width=self.calibration_cfg['conf_pool_width'],
                    discrete=False
                )
            # Get the label for the ith image.
            y_true = (context_labels[:, i, ...] > 0.5).long()
            # Update the pixel meters dict with the new support prediction.
            update_prob_pixel_meters(
                output_dict={
                    "y_probs": y_probs,
                    "y_true": y_true
                },
                calibration_cfg=self.calibration_cfg,
                pixel_level_records=support_pred_meter_dict 
            )
        # Get the statistics either from images or pixel meter dict.
        val_freqs = prob_bin_stats(
            pixel_meters_dict=support_pred_meter_dict, # Use the validation set stats.
            num_prob_bins=self.calibration_cfg['num_prob_bins'], # Use 15 bins
            device="cuda"
        )['bin_freqs'] # Bins
        # Softmax the logits to get probabilities
        uncal_target_probs = torch.sigmoid(target_logits)
        # Calculate the bin ownership map and transform the probs.
        prob_bin_ownership_map = get_bin_per_sample(
            pred_map=uncal_target_probs.squeeze(1),
            bin_starts=self.conf_bins,
            bin_widths=self.conf_bin_widths
        ).unsqueeze(1) # B x 1 x H x W
        # Get the new prob map by applying the calibration.
        return val_freqs[prob_bin_ownership_map]

    @property
    def device(self):
        return "cpu"