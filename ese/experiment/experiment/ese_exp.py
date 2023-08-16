# torch imports
import torch
from torch.utils.data import DataLoader

# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import
from ionpy.util.torchutils import to_device

# Misc imports
import matplotlib.pyplot as plt


class CalibrationExperiment(TrainExperiment):

    def build_data(self):
        data_cfg = self.config["data"].to_dict()
        dataset_cls = absolute_import(data_cfg.pop("_class"))

        dataset = data_cfg.pop("dataset")

        self.train_dataset = dataset_cls(dataset=dataset, split="train", **data_cfg)
        self.cal_dataset = dataset_cls(dataset=dataset, split="cal", **data_cfg)
        self.val_dataset = dataset_cls(dataset=dataset, split="val", **data_cfg)
    
    def build_dataloader(self):
        dl_cfg = self.config["dataloader"]

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_cfg)

    def run_step(self, batch_idx, batch, backward=True, augmentation=False, epoch=None, phase=None):

        # Send data and labels to device.
        x, y = to_device(batch, self.device)

        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline(x, y)

        yhat = self.model(x)
        loss = self.loss_func(yhat, y)

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

        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return {
            "loss": loss,
            "ytrue": y,
            "ypred": yhat,
            "batch_idx": batch_idx,
        }

    def run(self):
        super().run()
