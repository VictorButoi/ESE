import torch
import matplotlib.pyplot as plt


def ShowPredictions(experiment):
    def ShowPredictionsCallback(batch):
        x = batch["x"]
        y = batch["ytrue"]
        yhat = batch["ypred"]
        bs = x.shape[0]

        f, axarr = plt.subplots(bs, 4, figsize=(20, bs*5))
        for b_idx in range(bs):

            if bs == 1:
                axarr[0].set_title("Image")
                im1 = axarr[0].imshow(x[0, 0, :, :].cpu().numpy(), cmap="gray")
                f.colorbar(im1, ax=axarr[0], orientation='vertical')

                axarr[1].set_title("Seg")
                im2 = axarr[1].imshow(y[0, 0, :, :].cpu().numpy(), cmap="gray")
                f.colorbar(im2, ax=axarr[1], orientation='vertical')

                axarr[2].set_title("Pred")
                im3 = axarr[2].imshow(yhat[0, 0, :, :].cpu().detach().numpy(), cmap="gray")
                f.colorbar(im3, ax=axarr[2], orientation='vertical')

                axarr[3].set_title("Pred Sigmoid")
                im4 = axarr[3].imshow(torch.sigmoid(yhat[0, 0, :, :].cpu().detach()).numpy(), cmap="gray")
                f.colorbar(im4, ax=axarr[3], orientation='vertical')
            else:
                axarr[b_idx, 0].set_title("Image")
                im1 = axarr[b_idx, 0].imshow(x[b_idx, 0, :, :].cpu().numpy(), cmap="gray")
                f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

                axarr[b_idx, 1].set_title("Seg")
                im2 = axarr[b_idx, 1].imshow(y[b_idx, 0, :, :].cpu().numpy(), cmap="gray")
                f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

                axarr[b_idx, 2].set_title("Pred")
                im3 = axarr[b_idx, 2].imshow(yhat[b_idx, 0, :, :].cpu().detach().numpy(), cmap="gray")
                f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

                axarr[b_idx, 3].set_title("Pred Sigmoid")
                im4 = axarr[b_idx, 3].imshow(torch.sigmoid(yhat[b_idx, 0, :, :].cpu().detach()).numpy(), cmap="gray")
                f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

        plt.show()

    return ShowPredictionsCallback