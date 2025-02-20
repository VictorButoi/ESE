# torch imports
import torch
import torchvision.transforms as T
# misc imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, Optional, Literal
# local imports
from ..metrics.local_ps import bin_stats
from ..metrics.utils import get_bin_per_sample
from ..analysis.cal_plots.reliability_plots import reliability_diagram


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build_subj_dict(df, npz_dict):
    subj_dict = {}

    for data_id in df['data_id'].unique():
        subj_dict[data_id] = {}

        # Unique pretrained loss functions
        if 'pretrain_loss_fn' not in df.columns:
            unique_pt_loss_fns = [None]
        else:
            unique_pt_loss_fns = df['pretrain_loss_fn'].unique()

        for pre_loss_fn in unique_pt_loss_fns:
            # For each unique pretraining loss func
            if pre_loss_fn is None:
                loss_fn_df = df
                pre_loss_fn = 'None'
            else:
                loss_fn_df = df[df['pretrain_loss_fn'] == pre_loss_fn]

            subj_dict[data_id][pre_loss_fn] = {}

            for seed in loss_fn_df['experiment_pretrained_seed'].unique():

                e_seed_df = loss_fn_df[loss_fn_df['experiment_pretrained_seed'] == seed]
                subj_dict[data_id][pre_loss_fn][seed] = {}

                for e_tor in e_seed_df['estimator'].unique():

                    e_df = e_seed_df[e_seed_df['estimator'] == e_tor]
                    assert len(e_df['log_dir'].unique()) == 1

                    # Get the pred and put it in the dict.
                    log_dir = e_df['log_dir'].unique()[0]
                    pred_logits = npz_dict[log_dir][data_id].squeeze()
                    pred_probs = sigmoid(pred_logits)

                    # If this is a HARD estimator, then we need to threshold the predictions at the experiment_threshold
                    if 'LTS' in e_tor:
                        for est_mat in ['soft', 'hard']:
                            if est_mat == 'hard':
                                pred_probs = (pred_probs > 0.5)
                            # Place this in the dict.
                            subj_dict[data_id][pre_loss_fn][seed][f'LTS {est_mat}'] = pred_probs

                    else:
                        if "hard" in e_tor:
                            pred_probs = (pred_probs > 0.5)
                        # Place this in the dict.
                        subj_dict[data_id][pre_loss_fn][seed][e_tor] = pred_probs
    # Return the built dict.
    return subj_dict


def display_subj(
    data_id, 
    loss_func, 
    preds_dict, 
    dataset_obj, 
    seeds, 
    method_order,
    slicing: Literal["midslice","maxslice"] = "midslice",
    planes=None
):
    dat_dict = dataset_obj._db[data_id]
    img = dat_dict['img']
    lab = dat_dict['seg']
    seed_preds_dict = preds_dict[data_id][loss_func]
    # Display all the preds near each other. With room for the image and label.
    num_boxes = 2 + len(method_order) + 1 # Image, Label, and then all the methods, and the diff between soft and LTS.
    num_rows = len(seeds)
    dims = img.shape

    # Make little functions to display the images.
    def display_ims(plane=None):
        f, axarr = plt.subplots(num_rows, num_boxes, figsize=(4 * num_boxes, 3 * len(seeds)))
        # Make the background of the image white.
        f.patch.set_facecolor('white')
        if plane is not None:
            # IF the plane is defined, then we want to give a overall title to th 
            # subplots that says these are the predictions for axis = plane
            f.suptitle(f"Predictions for {data_id}, plane={plane}", fontsize=16)
            f.subplots_adjust(top=0.75)
            # Get the midslice for now.
            if slicing == "midslice":
                slice_idx = dims[plane] // 2
            else:
                all_dims = np.arange(len(dims))
                dims_to_take = np.delete(all_dims, plane)
                lab_per_slice = np.sum(lab, axis=tuple(dims_to_take))
                slice_idx = np.argmax(lab_per_slice)
            plane_img = img.take(slice_idx, axis=plane)
            plane_lab = lab.take(slice_idx, axis=plane)
        else:
            plane_img = img
            plane_lab = lab
        # Normalize the image and label to be between 0 and 1.
        plane_img = (plane_img - plane_img.min()) / (plane_img.max() - plane_img.min())
        plane_lab = (plane_lab - plane_lab.min()) / (plane_lab.max() - plane_lab.min())
        
        # We want to crop the image/lab to the region where there is brain + 5 pixels.
        # This is to make the images more interpretable.
        plane_shape = plane_img.shape
        min_y, min_x = 0, 0
        max_y, max_x = plane_shape[0], plane_shape[1]

        # Find indices where brain is present along x-axis, y-axis
        intensities_per_row = np.sum(plane_img, axis=1)
        min_row_intensity = intensities_per_row.min()
        indices_y = np.where(intensities_per_row > min_row_intensity)[0]

        intensities_per_col = np.sum(plane_img, axis=0)
        min_col_intensity = intensities_per_col.min()
        indices_x = np.where(intensities_per_col > min_col_intensity)[0]

        # If we have valid indices, do the crop! (with buffer of 5 pixels) 
        if indices_x.size > 0:
            # Calculate min and max x-values with padding, ensuring they stay within bounds
            min_x = max(indices_x.min() - 5, 0)
            max_x = min(indices_x.max() + 5, max_x)
        if indices_y.size > 0:
            # Calculate min and max y-values with padding, ensuring they stay within bounds
            min_y = max(indices_y.min() - 5, 0)
            max_y = min(indices_y.max() + 5, max_y)
        
        # Do a crop of the image and label.
        def crop_img(img, min_x, max_x, min_y, max_y):
            return img[min_y:max_y, min_x:max_x]

        # Crop the img and label to the boundary of the brain
        plane_img = crop_img(plane_img, min_x, max_x, min_y, max_y)
        plane_lab = crop_img(plane_lab, min_x, max_x, min_y, max_y)
            
        for s_idx, seed in enumerate(seeds):
            # Get the predictions corresponding to this seed.
            methods_dict = seed_preds_dict[seed]

            if num_rows == 1:
                im_ax = axarr[0].imshow(plane_img, cmap='gray')
                axarr[0].set_title(data_id)
                f.colorbar(im_ax, ax=axarr[0])

                lab_ax = axarr[1].imshow(plane_lab, cmap='gray')
                axarr[1].set_title("Label")
                f.colorbar(lab_ax, ax=axarr[1])

                assert set(method_order) == set(methods_dict.keys()),\
                    f"Got {set(method_order)} and {set(methods_dict.keys())}." 

                for m_idx, method in enumerate(method_order):
                    if plane is not None:
                        method_plane_pred = methods_dict[method].take(slice_idx, axis=plane)
                    else:
                        method_plane_pred = methods_dict[method]
                    # Crop the method plane pred.
                    method_plane_pred = crop_img(method_plane_pred, min_x, max_x, min_y, max_y)

                    meth_ax = axarr[m_idx + 2].imshow(method_plane_pred, cmap='gray')
                    axarr[m_idx + 2].set_title(method + f", seed={seed}")
                    f.colorbar(meth_ax, ax=axarr[m_idx + 2])


                # Custom looking at the dif from soft to LTS.  
                dif_img = (methods_dict["LTS soft"] - methods_dict["None soft"])
                if plane is not None:
                    dif_plane = dif_img.take(slice_idx, axis=plane)
                else: 
                    dif_plane = dif_img
                # Crop the dif image.
                dif_plane = crop_img(dif_plane, min_x, max_x, min_y, max_y)

                dif_plt = axarr[-1].imshow(dif_plane, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
                f.colorbar(dif_plt, ax=axarr[-1])
                axarr[-1].set_title(f"Diff, seed={seed}")
            else:
                im_ax = axarr[s_idx, 0].imshow(plane_img, cmap='gray')
                axarr[s_idx, 0].set_title("Image")
                f.colorbar(im_ax, ax=axarr[s_idx, 0])

                lab_ax = axarr[s_idx, 1].imshow(plane_lab, cmap='gray')
                axarr[s_idx, 1].set_title("Label")
                f.colorbar(lab_ax, ax=axarr[s_idx, 1])

                assert set(method_order) == set(methods_dict.keys())
                for m_idx, method in enumerate(method_order):
                    if plane is not None:
                        method_plane_pred = methods_dict[method].take(slice_idx, axis=plane)
                    else:
                        method_plane_pred = methods_dict[method]
                    # Crop the method plane pred.
                    method_plane_pred = crop_img(method_plane_pred, min_x, max_x, min_y, max_y)

                    meth_ax = axarr[s_idx, m_idx + 2].imshow(method_plane_pred, cmap='gray')
                    axarr[s_idx, m_idx + 2].set_title(method + f", seed={seed}")
                    f.colorbar(meth_ax, ax=axarr[s_idx, m_idx + 2])

                # Custom looking at the dif from soft to LTS.  
                dif_img = (methods_dict["LTS soft"] - methods_dict["None soft"])
                if plane is not None:
                    dif_plane = dif_img.take(slice_idx, axis=plane)
                else: 
                    dif_plane = dif_img
                # Crop the dif image.
                dif_plane = crop_img(dif_plane, min_x, max_x, min_y, max_y)

                dif_plt = axarr[s_idx, -1].imshow(dif_plane, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
                f.colorbar(dif_plt, ax=axarr[s_idx, -1])
                axarr[s_idx, -1].set_title(f"Diff, seed={seed}")
        # Turn off all of the ticks.
        for ax in axarr.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
    # Either iterate through the axes or display the images. 
    if planes:
        for plane in planes:
            display_ims(plane=plane)
            plt.show()
        print("--------------------------------------------------------------------------------")
    else:
        display_ims()


class ShowSegmentationPredictions:
    
    def __init__(
        self, 
        exp = None, 
        col_wrap: int = 4,
        threshold: float = 0.5,
        num_prob_bins: int = 15,
        size_per_image: int = 5,
        denormalize: Optional[Any] = None,
        temperature: Optional[float] = None
    ):
        self.col_wrap = col_wrap
        self.threshold = threshold
        self.size_per_image = size_per_image
        self.temperature = temperature
        self.num_prob_bins = num_prob_bins
        # Sometimes we normalize the intensity values so we need to denormalize them for visualization.
        if denormalize is not None:
            # Denormalization transform
            self.denormalize = T.Normalize(
                mean=[-m/s for m, s in zip(denormalize['mean'], denormalize['std'])],
                std=[1/s for s in denormalize['std']]
            )
        else:
            self.denormalize = None


    def __call__(self, batch):
        # If our pred has a different batchsize than our inputs, we
        # need to tile the input and label to match the batchsize of
        # the prediction.
        x = batch["x"]
        y = batch["y_true"]
        y_hat = batch["y_pred"]
        
        # Transfer image and label to the cpu.
        x = x.detach().cpu()
        y = y.detach().cpu() 

        # Get the predicted label
        y_hat = y_hat.detach().cpu()
        bs = x.shape[0]
        num_pred_classes = y_hat.shape[1]

        # Prints some metric stuff
        if "loss" in batch:
            print("Loss: ", batch["loss"].item())
        # If we are using a temperature, divide the logits by the temperature.
        if self.temperature is not None:
            y_hat = y_hat / self.temperature

        # Make a hard prediction.
        if num_pred_classes > 1:
            y_hat = torch.softmax(y_hat, dim=1)
            if num_pred_classes == 2 and self.threshold != 0.5:
                y_hard = (y_hat[:, 1, :, :] > self.threshold).int()
            else:
                y_hard = torch.argmax(y_hat, dim=1)
        else:
            y_hat = torch.sigmoid(y_hat)
            y_hard = (y_hat > self.threshold).int()

        # Keep the original y and y_hat so we can use them for the reliability diagrams.
        original_y = y
        original_y_hat = y_hat
        # If x is 5 dimensionsal, we are dealing with 3D data and we need to treat the volumes
        # slightly differently.
        if len(x.shape) == 5:
            # We want to look at the slice corresponding to the maximum amount of label.
            y_squeezed = y.squeeze(1) # (B, Spatial Dims)
            # Sum over the spatial dims that aren't the last one.
            lab_per_slice = y_squeezed.sum(dim=tuple(range(1, len(y_squeezed.shape) - 1)))
            # Get the max slices per batch item.
            max_slices = torch.argmax(lab_per_slice, dim=1)
            # Index into our 3D tensors with this.
            x = torch.stack([x[i, ...,  max_slices[i]] for i in range(bs)]) 
            y_hard = torch.stack([y_hard[i, ..., max_slices[i]] for i in range(bs)])
            #``
            # Get the max slice for the label.
            y = torch.stack([y[i, ..., max_slices[i]] for i in range(bs)])
            y_hat = torch.stack([y_hat[i, ..., max_slices[i]] for i in range(bs)])

        # Squeeze all tensors in prep.
        x = x.permute(0, 2, 3, 1).numpy().squeeze() # Move channel dimension to last.
        y = y.numpy().squeeze()
        y_hard = y_hard.numpy().squeeze()
        y_hat = y_hat.squeeze()

        # DETERMINE THE IMAGE CMAP
        if x.shape[-1] == 3:
            x = x.astype(int)
            img_cmap = None
        else:
            img_cmap = "gray"

        if num_pred_classes <= 2:
            label_cm = "gray"
        else:
            colors = [(0, 0, 0)] + [(np.random.random(), np.random.random(), np.random.random()) for _ in range(num_pred_classes - 1)]
            cmap_name = "seg_map"
            label_cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=num_pred_classes)

        if bs == 1:
            ncols = 7
        else:
            ncols = 4
        f, axarr = plt.subplots(nrows=bs, ncols=ncols, figsize=(ncols * self.size_per_image, bs*self.size_per_image))

        # Go through each item in the batch.
        for b_idx in range(bs):
            if bs == 1:
                axarr[0].set_title("Image")
                im1 = axarr[0].imshow(x, cmap=img_cmap, interpolation='None')
                f.colorbar(im1, ax=axarr[0], orientation='vertical')

                axarr[1].set_title("Label")
                im2 = axarr[1].imshow(y, cmap=label_cm, interpolation='None')
                f.colorbar(im2, ax=axarr[1], orientation='vertical')

                axarr[2].set_title("Hard Prediction")
                im3 = axarr[2].imshow(y_hard, cmap=label_cm, interpolation='None')
                f.colorbar(im3, ax=axarr[2], orientation='vertical')

                if len(y_hat.shape) == 3:
                    max_probs = torch.max(y_hat, dim=0)[0]
                    freq_map = (y_hard == y)
                else:
                    assert len(y_hat.shape) == 2, "Soft prediction must be 2D if not 3D."
                    max_probs = y_hat
                    freq_map = y

                axarr[3].set_title("Max Probs")
                im4 = axarr[3].imshow(max_probs, cmap='gray', vmin=0.0, vmax=1.0, interpolation='None')
                f.colorbar(im4, ax=axarr[3], orientation='vertical')

                axarr[4].set_title("Brier Map")
                im5 = axarr[4].imshow(
                    (max_probs - freq_map), 
                    cmap='RdBu_r', 
                    vmax=1.0, 
                    vmin=-1.0, 
                    interpolation='None')
                f.colorbar(im5, ax=axarr[4], orientation='vertical')

                miscal_map = np.zeros_like(max_probs)
                # Figure out where each pixel belongs (in confidence)
                toplabel_bin_ownership_map = get_bin_per_sample(
                    pred_map=max_probs[None],
                    n_spatial_dims=2,
                    class_wise=False,
                    num_prob_bins=self.num_prob_bins,
                    int_start=0.0,
                    int_end=1.0
                ).squeeze()
                # Fill the bin regions with the miscalibration.
                max_probs = max_probs.numpy()
                for bin_idx in range(self.num_prob_bins):
                    bin_mask = (toplabel_bin_ownership_map == bin_idx)
                    if bin_mask.sum() > 0:
                        miscal_map[bin_mask] = (max_probs[bin_mask] - freq_map[bin_mask]).mean()

                # Plot the miscalibration
                axarr[5].set_title("Miscalibration Map")
                im6 = axarr[5].imshow(
                    miscal_map, 
                    cmap='RdBu_r', 
                    vmax=0.2, 
                    vmin=-0.2, 
                    interpolation='None')
                f.colorbar(im6, ax=axarr[5], orientation='vertical')

                # Plot the reliability diagram for the binary case of the foreground.
                reliability_diagram(
                    calibration_info=bin_stats(
                        y_pred=original_y_hat,
                        y_true=original_y,
                        num_prob_bins=self.num_prob_bins
                    ),
                    title="Reliability Diagram",
                    num_prob_bins=self.num_prob_bins,
                    class_type="Binary",
                    plot_type="bar",
                    bar_color="blue",
                    ax=axarr[6]
                )

                # turn off the axis and grid
                for x_idx, ax in enumerate(axarr):
                    # Don't turn off the last axis
                    if x_idx != len(axarr) - 1:
                        # ax.axis('off')
                        ax.grid(False)
            else:
                axarr[b_idx, 0].set_title("Image")
                im1 = axarr[b_idx, 0].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
                f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

                axarr[b_idx, 1].set_title("Label")
                im2 = axarr[b_idx, 1].imshow(y[b_idx], cmap=label_cm, interpolation='None')
                f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

                axarr[b_idx, 2].set_title("Soft Prediction")
                im3 = axarr[b_idx, 2].imshow(y_hat[b_idx], cmap=label_cm, interpolation='None')
                f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

                axarr[b_idx, 3].set_title("Hard Prediction")
                im4 = axarr[b_idx, 3].imshow(y_hard[b_idx], cmap=label_cm, interpolation='None')
                f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

                # turn off the axis and grid
                for ax in axarr[b_idx]:
                    ax.axis('off')
                    ax.grid(False)
        plt.show()

