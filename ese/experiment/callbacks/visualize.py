import torch
import matplotlib.pyplot as plt


def ShowPredictions(
        experiment, 
        label_cmap='gray',
        show_soft_pred=False
        ):

    # Get the experiment config
    exp_config = experiment.config.to_dict() 

    def ShowPredictionsCallback(batch):
        # Move the channel dimension to the last dimension
        x = batch["x"]
        bs = x.shape[0]
        x = x.squeeze()
        x = x.permute(1, 2, 0)

        # Prepare the raw pred depending on multiclass or not by softmaxing and argmaxing over first dimension.
        yhat = batch["ypred"].squeeze()
        if len(yhat.shape) == 3:
            assert not show_soft_pred, "Can't show soft predictions for multiclass"
            yhat = torch.softmax(yhat, dim=0)
            yhat = torch.argmax(yhat, dim=0)
        else:
            yhat = torch.sigmoid(yhat)

        # Get the label 
        y = batch["ytrue"].squeeze()

        # Prepare the tensors for visualization
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        yhat = yhat.cpu().detach().numpy()

        num_cols = 4 if show_soft_pred else 3
        f, axarr = plt.subplots(nrows=bs, ncols=num_cols, figsize=(20, bs*5))

        # Go through each item in the batch.
        for b_idx in range(bs):

            if bs == 1:
                axarr[0].set_title("Image")
                print(x.shape)
                plt.hist(x.flatten())
                plt.show()
                im1 = axarr[0].imshow(x, interpolation='None')
                f.colorbar(im1, ax=axarr[0], orientation='vertical')

                axarr[1].set_title("Label")
                im2 = axarr[1].imshow(y, cmap=label_cmap, interpolation='None')
                f.colorbar(im2, ax=axarr[1], orientation='vertical')

                axarr[2].set_title("Prediction")
                im3 = axarr[2].imshow(yhat, cmap=label_cmap, interpolation='None')
                f.colorbar(im3, ax=axarr[2], orientation='vertical')

                if show_soft_pred:
                    axarr[3].set_title("Soft Prediction")
                    im4 = axarr[3].imshow(yhat, cmap="gray")
                    f.colorbar(im4, ax=axarr[3], orientation='vertical')
            else:
                axarr[b_idx, 0].set_title("Image")
                im1 = axarr[b_idx, 0].imshow(x[b_idx, ...], interpolation='None')
                f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

                axarr[b_idx, 1].set_title("Label")
                im2 = axarr[b_idx, 1].imshow(y[b_idx, ...], cmap=label_cmap, interpolation='None')
                f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

                axarr[b_idx, 2].set_title("Prediction")
                im3 = axarr[b_idx, 2].imshow(yhat[b_idx, ...], cmap=label_cmap, interpolation='None')
                f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

                if show_soft_pred:
                    axarr[b_idx, 3].set_title("Soft Prediction")
                    im4 = axarr[b_idx, 3].imshow(yhat[b_idx, ...], cmap="gray")
                    f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

        plt.show()

    return ShowPredictionsCallback