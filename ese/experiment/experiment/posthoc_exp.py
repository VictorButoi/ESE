# misc imports
import os
# local imports
from .utils import load_experiment, process_logits_map
# torch imports
import torch
from torch.utils.data import DataLoader
# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.datasets.cuda import CUDACachedDataset
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.nn.util import num_params
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.torchutils import to_device
from ionpy.analysis import ResultsLoader


class PostHocExperiment(TrainExperiment):

    def build_data(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        # Get the data and transforms we want to apply
        pretrained_data_cfg = self.pretrained_exp.config["data"].to_dict()
        # Update the old cfg with new cfg (if it exists).
        if "data" in self.config:
            pretrained_data_cfg.update(self.config["data"].to_dict())
        total_config["data"] = pretrained_data_cfg
        autosave(total_config, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_config)
        
        # Get the dataset class and build the transforms
        dataset_cls = absolute_import(pretrained_data_cfg.pop("_class"))
        if "cuda" in pretrained_data_cfg:
            assert pretrained_data_cfg["preload"], "If you want to cache the dataset on the GPU, you must preload it."
            cache_dsets_on_gpu = pretrained_data_cfg.pop("cuda")
        else:
            cache_dsets_on_gpu = False
        # Build the datasets, apply the transforms
        self.train_dataset = dataset_cls(split="val", **pretrained_data_cfg)
        self.val_dataset = dataset_cls(split="cal", **pretrained_data_cfg)
        # Optionally cache the datasets on the GPU.
        if cache_dsets_on_gpu:
            self.train_dataset = CUDACachedDataset(self.train_dataset)
            self.val_dataset = CUDACachedDataset(self.val_dataset)

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
        ###################
        # BUILD THE MODEL #
        ###################
        # Get the configs of the experiment
        load_exp_cfg = {
            "device": "cuda",
            "build_data": False, # Important, we might want to modify the data construction.
        }
        if "config.yml" in os.listdir(total_config['train']['pretrained_dir']):
            self.pretrained_exp = load_experiment(
                path=total_config['train']['pretrained_dir'],
                **load_exp_cfg
            )
        else:
            rs = ResultsLoader()
            dfc = rs.load_configs(
                total_config['train']['pretrained_dir'],
                properties=False,
            )
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(dfc),
                selection_metric=total_config['train']['pretrained_select_metric'],
                **load_exp_cfg
            )
        # Make sure we use the old experiment seed.
        old_exp_config = self.pretrained_exp.config.to_dict() 
        total_config['experiment'] = old_exp_config['experiment']
        # Set important things about the model.
        self.config = Config(total_config)
        # Set the base model to not get updates.
        self.base_model = self.pretrained_exp.model
        self.base_model.eval()
        ########################
        # BUILD THE CALIBRATOR #
        ########################
        self.model = eval_config(self.config["model"])
        self.model.weights_init()
        self.properties["num_params"] = num_params(self.model)
    
    def build_loss(self):
        self.loss_func = eval_config(self.config["loss_func"])
    
    def run_step(self, batch_idx, batch, backward, **kwargs):
        # Send data and labels to device.
        x, y = to_device(batch, self.device)
        # Forward pass
        with torch.no_grad():
            yhat = self.base_model(x)
        # Calibrate the predictions.
        yhat_cal = self.model(yhat, image=x)
        # Calculate the loss between the pred and original preds.
        loss = self.loss_func(yhat_cal, y)
        # If backward then backprop the gradients.
        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        # Run step-wise callbacks if you have them.
        forward_batch = {
            "x": x,
            "ytrue": y,
            "ypred": yhat_cal,
            "loss": loss,
            "batch_idx": batch_idx,
        }
        self.run_callbacks("step", batch=forward_batch)
        return forward_batch

    def predict(self, 
                x, 
                multi_class,
                threshold=0.5):
        assert x.shape[0] == 1, "Batch size must be 1 for prediction for now."
        # Predict with the base model.
        with torch.no_grad():
            yhat = self.base_model(x)
        # Apply post-hoc calibration.
        logit_map = self.model(yhat, image=x) 
        # Get the hard prediction and probabilities
        prob_map, pred_map = process_logits_map(
            logit_map, 
            multi_class=multi_class, 
            threshold=threshold
            )
        # Return the outputs probs and predicted label map.
        return prob_map, pred_map

    def run(self):
        super().run()
