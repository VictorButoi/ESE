
# local imports
from .utils import process_pred_map
# torch imports
# IonPy imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import eval_config
from ionpy.nn.util import num_params
from ionpy.util import Config
from ionpy.util.torchutils import to_device
# misc imports
import einops


class CalibrationExperiment(TrainExperiment):

    def build_model(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        # Get the model and data configs.
        model_config = total_config["model"]
        data_config = total_config["data"]
        # transfer the arguments to the model config.
        if "in_channels" in data_config:
            in_channels = data_config.pop("in_channels")
            out_channels = data_config.pop("out_channels")
            assert out_channels > 1, "Must be multi-class segmentation!"
            model_config["in_channels"] = in_channels
            model_config["out_channels"] = out_channels 
        # Set important things about the model.
        self.config = Config(total_config)
        self.model = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)
    
    def run_step(self, batch_idx, batch, backward, **kwargs):
        # Send data and labels to device.
        batch = to_device(batch, self.device)
        # Get the image and label.
        x, y = batch["img"], batch["label"]

        # For volume datasets, sometimes want to treat different slices as a batch.
        if ("slice_batch_size" in self.config["data"]) and (self.config["data"]["slice_batch_size"] > 1):
            assert x.shape[0] == 1, "Batch size must be 1 for slice batching."
            # This lets you potentially use multiple slices from 3D volumes by mixing them into a big batch.
            x = einops.rearrange(x, "b c h w -> (b c) 1 h w")
            y = einops.rearrange(y, "b c h w -> (b c) 1 h w")

        # torch.cuda.synchronize()
        # start = time.time()
        yhat = self.model(x)
        # torch.cuda.synchronize()
        # end = time.time()
        # print("Model forward pass time: ", end - start)

        if yhat.shape[1] > 1:
            y = y.long()

        loss = self.loss_func(yhat, y)
        # If backward then backprop the gradients.
        if backward:
            # torch.cuda.synchronize()
            # start = time.time()
            loss.backward()
            # torch.cuda.synchronize()
            # end = time.time()
            # print("Backward pass time: ", end - start)
            # print()
            self.optim.step()
            self.optim.zero_grad()
        # Run step-wise callbacks if you have them.
        forward_batch = {
            "x": x,
            "y_true": y,
            "y_pred": yhat,
            "loss": loss,
            "batch_idx": batch_idx,
        }
        self.run_callbacks("step", batch=forward_batch)
        return forward_batch
    
    def predict(self, 
                x, 
                multi_class,
                threshold=0.5,
                return_logits=False):
        assert x.shape[0] == 1, "Batch size must be 1 for prediction for now."
        # Get the label predictions
        logit_map = self.model(x) 
        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            logit_map, 
            multi_class=multi_class, 
            threshold=threshold,
            return_logits=return_logits
            )
        # Return the outputs
        return {
            'y_pred': prob_map, 
            'y_hard': pred_map 
        }