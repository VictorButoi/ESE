import torch
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def ShowPredictionsCallback(
    batch, 
    threshold: float = 0.5,
    size_per_iamge: int = 5,
):
    # If our pred has a different batchsize than our inputs, we
    # need to tile the input and label to match the batchsize of
    # the prediction.
    if ("y_probs" in batch) and (batch["y_probs"] is not None):
        pred_cls = "y_probs"
    else:
        assert ("y_logits" in batch) and (batch["y_logits"] is not None), "Must provide either probs or logits."
        pred_cls = "y_logits"

    if batch["x"].shape[0] != batch[pred_cls].shape[0]:
        assert batch["x"].shape[0] == 1, "Batchsize of input image must be 1 if batchsize of prediction is not 1."
        assert batch["y_true"].shape[0] == 1, "Batchsize of input label must be 1 if batchsize of prediction is not 1."
        bs = batch[pred_cls].shape[0]
        x = batch["x"].repeat(bs, 1, 1, 1)
        y = batch["y_true"].repeat(bs, 1, 1, 1)
    else:
        x = batch["x"]
        y = batch["y_true"]
    
    # Transfer image and label to the cpu.
    x = x.detach().cpu().permute(0, 2, 3, 1) # Move channel dimension to last.
    y = y.detach().cpu() 

    # Get the predicted label
    y_hat = batch[pred_cls].detach().cpu()
    bs = x.shape[0]
    num_pred_classes = y_hat.shape[1]

    if num_pred_classes <= 2:
        label_cm = "gray"
    else:
        colors = [(0, 0, 0)] + [(np.random.random(), np.random.random(), np.random.random()) for _ in range(num_pred_classes - 1)]
        cmap_name = "seg_map"
        label_cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=num_pred_classes)

    # Prints some metric stuff
    if "loss" in batch:
        print("Loss: ", batch["loss"].item())

    # If x is rgb
    if x.shape[-1] == 3:
        x = x.int()
        img_cmap = None
    else:
        img_cmap = "gray"

    if num_pred_classes > 1:
        if pred_cls == "y_logits":
            y_hat = torch.softmax(y_hat, dim=1)
        y_hard = torch.argmax(y_hat, dim=1)
    else:
        if pred_cls == "y_logits":
            y_hat = torch.sigmoid(y_hat)
        y_hard = (y_hat > threshold).int()
    
    # Squeeze all tensors in prep.
    x = x.numpy().squeeze()
    y = y.numpy().squeeze()
    y_hard = y_hard.numpy().squeeze()
    y_hat = y_hat.squeeze()

    # num_cols = 5 if (softpred_dim is not None) else 3
    f, axarr = plt.subplots(nrows=bs, ncols=5, figsize=(5 * size_per_iamge, bs*size_per_iamge))

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
            else:
                assert len(y_hat.shape) == 2, "Soft prediction must be 2D if not 3D."
                max_probs = y_hat

            axarr[3].set_title("Max Probs")
            im4 = axarr[3].imshow(max_probs, cmap='gray', vmin=0.0, vmax=1.0, interpolation='None')
            f.colorbar(im4, ax=axarr[3], orientation='vertical')

            pix_accuracy = (y_hard == y)
            axarr[4].set_title("Pixel Miscalibration")
            im5 = axarr[4].imshow(
                (max_probs - pix_accuracy), 
                cmap='RdBu_r', 
                vmax=1.0, 
                vmin=-1.0, 
                interpolation='None')
            f.colorbar(im5, ax=axarr[4], orientation='vertical')

            # turn off the axis and grid
            for ax in axarr:
                ax.axis('off')
                ax.grid(False)
        else:
            axarr[b_idx, 0].set_title("Image")
            im1 = axarr[b_idx, 0].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

            axarr[b_idx, 1].set_title("Label")
            im2 = axarr[b_idx, 1].imshow(y[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

            axarr[b_idx, 2].set_title("Hard Prediction")
            im3 = axarr[b_idx, 2].imshow(y_hard[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

            if len(y_hat.shape) == 4:
                max_probs = torch.max(y_hat, dim=1)[0]
            else:
                assert len(y_hat.shape) == 3, "Soft prediction must be 2D if not 3D."
                max_probs = y_hat

            axarr[b_idx, 3].set_title("Max Probs")
            im4 = axarr[b_idx, 3].imshow(max_probs[b_idx], cmap='gray', vmin=0.0, vmax=1.0, interpolation='None')
            f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

            axarr[b_idx, 4].set_title("Pixel Miscalibration")
            pix_accuracy = (y_hard == y)
            im5 = axarr[b_idx, 4].imshow(
                (max_probs[b_idx] - pix_accuracy[b_idx]), 
                cmap='RdBu_r', 
                vmax=1.0, 
                vmin=-1.0, 
                interpolation='None')
            f.colorbar(im5, ax=axarr[b_idx, 4], orientation='vertical')

            # turn off the axis and grid
            for ax in axarr[b_idx]:
                ax.axis('off')
                ax.grid(False)
    plt.show()


def ShowPredictions(
        experiment
        ):
    return ShowPredictionsCallback