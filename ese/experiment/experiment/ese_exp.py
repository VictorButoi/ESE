# misc imports
import pathlib

# torch imports
import torch

# diseg imports
from diseg.experiment.datasets import ModifiedCUDACachedDataset
from diseg.experiment.augmentation import augmentations_from_config

# IonPy imports
from IonPy.experiment import TrainExperiment
from IonPy.experiment.util import absolute_import, eval_config
from IonPy.nn.util import num_params
from IonPy.util.hash import json_digest


class DisegExperiment(TrainExperiment):

    def build_data(self):
        data_cfg = self.config["data"].to_dict()
        
        # Grab the size and the class before initializing the dataset.
        dataset_constructor = data_cfg.pop("_class", None) or data_cfg.pop("_fn")
        dataset_cls = absolute_import(dataset_constructor)

        # The source dataset is always the training split. We validate on the test
        # just so that we can see generalization.
        self.source_dataset = ModifiedCUDACachedDataset(dataset_cls(split="train", **data_cfg), **data_cfg)
        
        # Training on learning how to sample for validation split and then validating on test split
        # Note that this is not the same as training on the training split and validating on the test split.
        self.train_dataset = ModifiedCUDACachedDataset(dataset_cls(split="val", **data_cfg))
        self.val_dataset = ModifiedCUDACachedDataset(dataset_cls(split="test", **data_cfg))
        
    def build_augmentations(self):
        if "augmentations" in self.config:
            aug_cfg = self.config.to_dict()["augmentations"]
            self.aug_pipeline = augmentations_from_config(aug_cfg)
            # Taken from UniverSeg code, unsure why 8 is used
            self.properties["aug_digest"] = json_digest(self.config["augmentations"])[:8]

    def build_model(self):
        # Initialize the model and initialize a sampler for the model.
        self.model = eval_config(self.config["model"])
        self.model.__init_sampler__(self.config["sampler"], dset_size=len(self.source_dataset))

        # Keep track of the number of parameters of the sampling_model.
        self.properties["num_params"] = num_params(self.model)

        # load the in-context learning model and freeze the parameters
        ic_modal_class = absolute_import(self.config["incontext_network._class"])
        ic_experiment = ic_modal_class(pathlib.Path(self.config["incontext_network.load_dir"]))
        ic_experiment.load(self.config["incontext_network.load_epoch"])
        ic_experiment.to_device()

        # Freeze the parameters of the IC model
        self.ic_model = ic_experiment.model
        for _, p in self.ic_model.named_parameters():
           p.requires_grad = False
    
    def build_loss(self):
        super().build_loss()
        if self.config.get("train.fp16", False):
            assert torch.cuda.is_available()
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def run_step(self, *, batch_idx, batch, backward=True, augmentation=None, epoch=None, **kwargs):

        # Unpack the batch, which will already be on the GPU
        x, y, subj_idx = batch

        # Augment the query image to reduce overfitting.
        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline(x, y)

        # Sample the support set from the source dataset
        support_image_sets, support_label_sets, logits, support_inds = self.model(x, y, self.source_dataset, subj_idx=subj_idx, epoch=epoch, backward=backward)

        # Copy the query image multiple times in the batch dimension
        query_images = x.repeat(self.config["sampler.batch_size"], 1, 1, 1)
        query_labels = y.repeat(self.config["sampler.batch_size"], 1, 1, 1)

        # Make predictions for supports sampled with the frozen IC model
        yhat, _ = self.ic_model(support_image_sets, support_label_sets, query_images)
        loss = self.loss_func(yhat, query_labels, logits=logits, support_inds=support_inds)

        # Add the L1 regularization term to sparsify the logits
        loss = loss + (self.config["train.l1_lambda"] * torch.norm(logits, p=1))

        if backward:
            loss.backward()

            # Perform the optimization step only after a certain number of backward passes
            if (batch_idx+1) % self.config["train.accumulate_steps"] == 0:  
                self.optim.step()
                self.optim.zero_grad()

        return {"loss": loss, "ytrue": query_labels, "ypred": yhat}

    def run(self):
        super().run()
