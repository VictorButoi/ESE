# torch imports
import torch
from torch.utils.data import DataLoader

# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import
from ionpy.util.torchutils import to_device


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

        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return {
            "loss": loss,
            "x": x,
            "ytrue": y,
            "ypred": yhat,
            "batch_idx": batch_idx,
        }

    def run(self):
        super().run()
