# local imports
from .ese_exp import CalibrationExperiment
# torch imports
from torch.utils.data import DataLoader
# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import eval_config
from ionpy.nn.util import num_params
from ionpy.util import Config
from ionpy.util.torchutils import to_device
from ionpy.analysis import ResultsLoader


class PostHocExperiment(TrainExperiment):

    def build_data(self):
        # Build the datasets, apply the transforms
        self.train_dataset = self.pretrained_exp.cal_dataset
        self.val_dataset = self.pretrained_exp.val_dataset
    
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
        ###################
        # BUILD THE MODEL #
        ###################
        # Get the configs of the experiment
        rs = ResultsLoader()
        dfc = rs.load_configs(
            total_config['train']['pretrained_dir'],
            properties=False,
        )
        self.pretrained_exp = rs.get_best_experiment(
            df=rs.load_metrics(dfc),
            exp_class=CalibrationExperiment,
            device="cuda"
        )
        self.base_model = self.pretrained_exp.model
        self.base_model.eval()
        ########################
        # BUILD THE CALIBRATOR #
        ########################
        self.model = eval_config(self.config["model"])
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
