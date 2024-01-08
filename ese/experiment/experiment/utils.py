import torch
import einops
import json
import pathlib
from typing import Any, Optional
# ionpy imports
from ionpy.experiment.util import absolute_import


def process_logits_map(conf_map, multi_class, threshold=0.5):
    # Dealing with multi-class segmentation.
    if conf_map.shape[1] > 1:
        conf_map = torch.softmax(conf_map, dim=1)
        # Add back the channel dimension (1)
        pred_map = torch.argmax(conf_map, dim=1)
        pred_map = einops.rearrange(pred_map, "b h w -> b 1 h w")
    else:
        # Get the prediction
        conf_map = torch.sigmoid(conf_map) # Note: This might be a bug for bigger batch-sizes.
        pred_map = (conf_map >= threshold).float()
        if multi_class:
            conf_map = torch.max(torch.cat([1 - conf_map, conf_map], dim=1), dim=1)[0]
            # Add back the channel dimension (1)
            conf_map = einops.rearrange(conf_map, "b h w -> b 1 h w")
    # Return the outputs probs and predicted label map.
    return conf_map, pred_map


def load_experiment(
    device="cuda",
    checkpoint="max-val-dice_score",
    build_data=True,
    df: Optional[Any] = None, 
    path: Optional[str] = None,
    selection_metric: Optional[str] = None,
):
    if path is None:
        assert selection_metric is not None, "Must provide a selection metric if no path is provided."
        assert df is not None, "Must provide a dataframe if no path is provided."
        phase, score = selection_metric.split("-")
        subdf = df.select(phase=phase)
        sorted_df = subdf.sort_values(score, ascending=False)
        exp_path = sorted_df.iloc[0].path
    else:
        exp_path = path
    # Get the experiment class
    properties_dir = pathlib.Path(exp_path) / "properties.json"
    with open(properties_dir, 'r') as prop_file:
        props = json.load(prop_file)
    exp_class = absolute_import(f'ese.experiment.experiment.{props["experiment"]["class"]}')
    # Load the experiment
    loaded_exp = exp_class(exp_path, build_data=build_data)
    if checkpoint is not None:
        loaded_exp.load(tag=checkpoint)
    # Set the device
    loaded_exp.device = torch.device(device)
    if device == "cuda":
        loaded_exp.to_device()
    # # Place the logs in the experiment, will be hand later
    # loaded_exp.logs = df.select(path=exp_path).reset_index(drop=True)
    # Return the modified loaded exp.
    return loaded_exp