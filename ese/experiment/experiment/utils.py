import torch
import einops
import json
from pathlib import Path
from typing import Any, Optional
from pydantic import validate_arguments
# ionpy imports
from ionpy.experiment.util import absolute_import


def parse_class_name(class_name):
    return class_name.split("'")[-2]


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def process_pred_map(
    conf_map: torch.Tensor, 
    multi_class: bool, 
    threshold: float = 0.5, 
    from_logits: bool = True,
    return_logits: bool = False # in the case we just want to pass through.
    ):
    if return_logits:
        return conf_map, None
    else:
        # Dealing with multi-class segmentation.
        if conf_map.shape[1] > 1:
            # Get the probabilities
            if from_logits:
                conf_map = torch.softmax(conf_map, dim=1)
            # Add back the channel dimension (1)
            pred_map = torch.argmax(conf_map, dim=1).unsqueeze(1)
        else:
            # Get the prediction
            if from_logits:
                conf_map = torch.sigmoid(conf_map) # Note: This might be a bug for bigger batch-sizes.
            pred_map = (conf_map >= threshold).float()
            if multi_class:
                conf_map = torch.max(torch.cat([1 - conf_map, conf_map], dim=1), dim=1)[0].unsqueeze(1)
        # Return the outputs probs and predicted label map.
        return conf_map, pred_map


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_experiment(
    device="cuda",
    checkpoint="max-val-dice_score",
    load_data=True,
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
    properties_dir = Path(exp_path) / "properties.json"
    with open(properties_dir, 'r') as prop_file:
        props = json.loads(prop_file.read())
    exp_class = absolute_import(f'ese.experiment.experiment.{props["experiment"]["class"]}')

    # Load the experiment
    loaded_exp = exp_class(exp_path, load_data=load_data)
    if checkpoint is not None:
        loaded_exp.load(tag=checkpoint)
    
    # Set the device
    loaded_exp.device = torch.device(device)
    if device == "cuda":
        loaded_exp.to_device()
    
    return loaded_exp