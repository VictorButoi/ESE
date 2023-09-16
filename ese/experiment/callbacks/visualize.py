import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def ShowPredictions(
        experiment, 
        show_soft_pred=False
        ):

    # Get the experiment config
    exp_config = experiment.config.to_dict() 

    # Generate a list of random colors, starting with black for background
    num_pred_classes = exp_config['model']['out_channels']

    if num_pred_classes == 2:
        colors = [(0, 0, 0), (1, 1, 1)]
    else:
        colors = [(0, 0, 0)] + [(np.random.random(), np.random.random(), np.random.random()) for _ in range(num_pred_classes - 1)]

    cmap_name = "seg_map"
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=num_pred_classes)

    def ShowPredictionsCallback(batch):
        # Move the channel dimension to the last dimension
        x = batch["x"].detach().cpu()
        x = x.permute(0, 2, 3, 1)

        # Get the true label
        y = batch["ytrue"].detach().cpu()

        # Get the predicted label
        yhat = batch["ypred"].detach().cpu()
        bs = x.shape[0]
        num_pred_classes = yhat.shape[1]

        if num_pred_classes > 1:
            assert not show_soft_pred, "Can't show soft predictions for multiclass"
            yhat = torch.argmax(torch.softmax(yhat, dim=1), dim=1)
            x = x.numpy().astype(np.uint8)
        else:
            x = x.numpy()
            yhat = torch.sigmoid(yhat)
        
        num_cols = 4 if show_soft_pred else 3
        f, axarr = plt.subplots(nrows=bs, ncols=num_cols, figsize=(20, bs*5))

        # Prepare the tensors
        y = y.numpy()
        yhat = yhat.numpy()

        # Go through each item in the batch.
        for b_idx in range(bs):

            if bs == 1:
                axarr[0].set_title("Image")
                im1 = axarr[0].imshow(x.squeeze(), interpolation='None')
                f.colorbar(im1, ax=axarr[0], orientation='vertical')

                axarr[1].set_title("Label")
                im2 = axarr[1].imshow(y.squeeze(), cmap=cm, interpolation='None')
                f.colorbar(im2, ax=axarr[1], orientation='vertical')

                axarr[2].set_title("Prediction")
                im3 = axarr[2].imshow(yhat.squeeze(), cmap=cm, interpolation='None')
                f.colorbar(im3, ax=axarr[2], orientation='vertical')

                if show_soft_pred:
                    axarr[3].set_title("Soft Prediction")
                    im4 = axarr[3].imshow(yhat.squeeze(), cmap="gray")
                    f.colorbar(im4, ax=axarr[3], orientation='vertical')
            else:
                axarr[b_idx, 0].set_title("Image")
                im1 = axarr[b_idx, 0].imshow(x[b_idx].squeeze(), interpolation='None')
                f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

                axarr[b_idx, 1].set_title("Label")
                im2 = axarr[b_idx, 1].imshow(y[b_idx].squeeze(), cmap=cm, interpolation='None')
                f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

                axarr[b_idx, 2].set_title("Prediction")
                im3 = axarr[b_idx, 2].imshow(yhat[b_idx].squeeze(), cmap=cm, interpolation='None')
                f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

                if show_soft_pred:
                    axarr[b_idx, 3].set_title("Soft Prediction")
                    im4 = axarr[b_idx, 3].imshow(yhat[b_idx, ...], cmap="gray")
                    f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

        plt.show()

    return ShowPredictionsCallback