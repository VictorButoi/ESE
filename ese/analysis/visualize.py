import numpy as np
from typing import Literal
import matplotlib.pyplot as plt


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