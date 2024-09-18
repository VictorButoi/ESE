import numpy as np
from typing import Literal
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build_subj_dict(df, npz_dict):
    subj_dict = {}

    for data_id in df['data_id'].unique():
        subj_dict[data_id] = {}

        for pre_loss_fn in df['pretrain_loss_fn'].unique():
            
            # For each unique pretraining loss func
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
            
        for s_idx, seed in enumerate(seeds):
            # Get the predictions corresponding to this seed.
            methods_dict = seed_preds_dict[seed]

            if num_rows == 1:
                axarr[0].imshow(plane_img, cmap='gray')
                axarr[0].set_title(data_id)
                axarr[1].imshow(plane_lab, cmap='gray')
                axarr[1].set_title("Label")
                assert set(method_order) == set(methods_dict.keys())
                for m_idx, method in enumerate(method_order):
                    if plane is not None:
                        method_plane_pred = methods_dict[method].take(slice_idx, axis=plane)
                    else:
                        method_plane_pred = methods_dict[method]
                    axarr[m_idx + 2].imshow(method_plane_pred, cmap='gray')
                    axarr[m_idx + 2].set_title(method + f", seed={seed}")
                # Custom looking at the dif from soft to LTS.  
                dif_img = (methods_dict["LTS soft"] - methods_dict["None soft"])
                if plane is not None:
                    dif_plane = dif_img.take(slice_idx, axis=plane)
                else: 
                    dif_plane = dif_img
                dif_plt = axarr[-1].imshow(dif_plane)
                f.colorbar(dif_plt, ax=axarr[-1])
                axarr[-1].set_title(f"Diff, seed={seed}")
            else:
                axarr[s_idx, 0].imshow(plane_img, cmap='gray')
                axarr[s_idx, 0].set_title("Image")
                axarr[s_idx, 1].imshow(plane_lab, cmap='gray')
                axarr[s_idx, 1].set_title("Label")
                assert set(method_order) == set(methods_dict.keys())
                for m_idx, method in enumerate(method_order):
                    if plane is not None:
                        method_plane_pred = methods_dict[method].take(slice_idx, axis=plane)
                    else:
                        method_plane_pred = methods_dict[method]
                    axarr[s_idx, m_idx + 2].imshow(method_plane_pred, cmap='gray')
                    axarr[s_idx, m_idx + 2].set_title(method + f", seed={seed}")
                # Custom looking at the dif from soft to LTS.  
                dif_img = (methods_dict["LTS soft"] - methods_dict["None soft"])
                if plane is not None:
                    dif_plane = dif_img.take(slice_idx, axis=plane)
                else: 
                    dif_plane = dif_img
                dif_plt = axarr[s_idx, -1].imshow(dif_plane)
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
    else:
        display_ims()