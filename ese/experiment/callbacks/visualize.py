import torch
import matplotlib.pyplot as plt


def ShowPredictions(experiment):
    def ShowPredictionsCallback(x, y, yhat):
        f, axarr = plt.subplots(1, 4, figsize=(20, 5))

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
        plt.show()

    return ShowPredictionsCallback