# torch imports
from torch.utils.data import DataLoader
# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.nn.util import num_params
from ionpy.util import Config
from ionpy.util.torchutils import to_device


class PostHocExperiment(TrainExperiment):

    def build_data(self):
        # Get the data and transforms we want to apply
        data_cfg = self.config["data"].to_dict()
        # Get the dataset class and build the transforms
        dataset_cls = absolute_import(data_cfg.pop("_class"))
        # Build the datasets, apply the transforms
        self.train_dataset = dataset_cls(split="train", **data_cfg)
        self.val_dataset = dataset_cls(split="val", **data_cfg)
        self.cal_dataset = dataset_cls(split="cal", **data_cfg)
    
    def build_dataloader(self, batch_size=None):
        # If the datasets aren't built, build them
        if not hasattr(self, "train_dataset"):
            self.build_data()
        dl_cfg = self.config["dataloader"].to_dict()

        # Optionally manually set the batch size.
        if batch_size is not None:
            dl_cfg["batch_size"] =  batch_size

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_cfg)

    def build_model(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        # Set important things about the model.
        self.config = Config(total_config)
        self.base_model = None
        self.calibrator = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)
    
    def build_loss(self):
        self.loss_func = eval_config(self.config["loss_func"])
    
    def run_step(self, batch_idx, batch, backward, **kwargs):
        # Send data and labels to device.
        x, y = to_device(batch, self.device)
        # Forward pass
        yhat = self.base_model(x)
        # Calculate the loss between the pred and original preds.
        loss = self.loss_func(yhat, y)
        # If backward then backprop the gradients.
        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        # Run step-wise callbacks if you have them.
        forward_batch = {
            "x": x,
            "ytrue": y,
            "ypred": yhat,
            "loss": loss,
            "batch_idx": batch_idx,
        }
        self.run_callbacks("step", batch=forward_batch)
        return forward_batch

    def run(self):
        super().run()
