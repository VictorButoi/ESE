import torch
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def ShowPredictionsCallback(
        batch, 
        from_logits: bool = True,
        threshold: float = 0.5,
        size_per_iamge: int = 3,
        softpred_dim: Optional[int] = None 
        ):
    # If our pred has a different batchsize than our inputs, we
    # need to tile the input and label to match the batchsize of
    # the prediction.
    if batch["x"].shape[0] != batch["ypred"].shape[0]:
        assert batch["x"].shape[0] == 1, "Batchsize of input image must be 1 if batchsize of prediction is not 1."
        assert batch["ytrue"].shape[0] == 1, "Batchsize of input label must be 1 if batchsize of prediction is not 1."
        bs = batch["ypred"].shape[0]
        batch["x"] = batch["x"].repeat(bs, 1, 1, 1)
        batch["ytrue"] = batch["ytrue"].repeat(bs, 1, 1, 1)
    
    # Move the channel dimension to the last dimension
    x = batch["x"].detach().cpu()
    x = x.permute(0, 2, 3, 1)
    y = batch["ytrue"].detach().cpu().numpy()

    # Get the predicted label
    yhat = batch["ypred"].detach().cpu()
    bs = x.shape[0]
    num_pred_classes = yhat.shape[1]

    if num_pred_classes == 2:
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
        x = x.numpy().astype(np.uint8)
        img_cmap = None
    else:
        x = x.numpy()
        img_cmap = "gray"

    if num_pred_classes > 1:
        if from_logits:
            yhat = torch.softmax(yhat, dim=1)
        yhard = torch.argmax(yhat, dim=1).numpy()
    else:
        if from_logits:
            yhat = torch.sigmoid(yhat)
        yhard = (yhat > threshold).numpy()
    
    num_cols = 4 if (softpred_dim is not None) else 3
    f, axarr = plt.subplots(nrows=bs, ncols=num_cols, figsize=(4 * size_per_iamge, bs*size_per_iamge))

    # Go through each item in the batch.
    for b_idx in range(bs):
        if bs == 1:
            axarr[0].set_title("Image")
            im1 = axarr[0].imshow(x.squeeze(), cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[0], orientation='vertical')

            axarr[1].set_title("Label")
            im2 = axarr[1].imshow(y.squeeze(), cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[1], orientation='vertical')

            axarr[2].set_title("Hard Prediction")
            im3 = axarr[2].imshow(yhard.squeeze(), cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[2], orientation='vertical')

            if softpred_dim is not None:
                axarr[3].set_title("Soft Prediction")
                im4 = axarr[3].imshow(yhat[softpred_dim].squeeze(), cmap=label_cm, interpolation='None')
                f.colorbar(im4, ax=axarr[3], orientation='vertical')

            # turn off the axis and grid
            for ax in axarr:
                ax.axis('off')
                ax.grid(False)
        else:
            axarr[b_idx, 0].set_title("Image")
            im1 = axarr[b_idx, 0].imshow(x[b_idx].squeeze(), interpolation='None')
            f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

            axarr[b_idx, 1].set_title("Label")
            im2 = axarr[b_idx, 1].imshow(y[b_idx].squeeze(), cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

            axarr[b_idx, 2].set_title("Hard Prediction")
            im3 = axarr[b_idx, 2].imshow(yhard[b_idx].squeeze(), cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

            if softpred_dim is not None:
                axarr[b_idx, 3].set_title("Soft Prediction")
                im4 = axarr[b_idx, 3].imshow(yhat[b_idx, softpred_dim].squeeze(), cmap=label_cm, interpolation='None')
                f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

            # turn off the axis and grid
            for ax in axarr[b_idx]:
                ax.axis('off')
                ax.grid(False)
    plt.show()


def ShowPredictions(
        experiment
        ):
    return ShowPredictionsCallback