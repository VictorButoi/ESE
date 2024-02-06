# torch imports
import torch
# ionpy imports
from ionpy.experiment.util import absolute_import
# misc imports
import json
import einops
from pathlib import Path
from typing import Any, Optional
from pydantic import validate_arguments
# local imports
from ..callbacks.visualize import ShowPredictionsCallback
from ..models.ensemble_utils import get_combine_fn


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
def reduce_ensemble_preds(
    output_dict: dict, 
    inference_cfg: dict
) -> dict:
    # Combine the outputs of the models.
    # NOTE: This will always do a softmax.
    ensemble_prob_map = get_combine_fn(inference_cfg["model"]["ensemble_cfg"][0])(
        output_dict["y_pred"], 
        combine_quantity=inference_cfg["model"]["ensemble_cfg"][1],
        weights=output_dict['ens_weights']
    )
    # Get the hard prediction and probabilities, if we are doing identity,
    # then we don't want to return probs.
    ensemble_prob_map, ensemble_pred_map = process_pred_map(
        ensemble_prob_map, 
        multi_class=True, 
        threshold=0.5,
        from_logits=False, # Ensemble methods already return probs.
        )
    return {
        "y_pred": ensemble_prob_map, # (B, C, H, W)
        "y_hard": ensemble_pred_map # (B, C, H, W)
    }

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_experiment(
    device: str = "cuda",
    checkpoint: str = "max-val-dice_score",
    load_data: bool = True,
    df: Optional[Any] = None, 
    path: Optional[str] = None,
    selection_metric: Optional[str] = None,
    exp_class: Optional[str] = None
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

    # Load the experiment
    if exp_class is None:
        # Get the experiment class
        properties_dir = Path(exp_path) / "properties.json"
        with open(properties_dir, 'r') as prop_file:
            props = json.loads(prop_file.read())
        exp_class = props["experiment"]["class"]
    # Load the class
    exp_class = absolute_import(f'ese.experiment.experiment.{exp_class}')
    loaded_exp = exp_class(exp_path, load_data=load_data)

    # Load the experiment
    if checkpoint is not None:
        loaded_exp.load(tag=checkpoint)
    
    # Set the device
    loaded_exp.device = torch.device(device)
    if device == "cuda":
        loaded_exp.to_device()
    
    return loaded_exp


def show_inference_examples(
    output_dict: dict,
    inference_cfg: dict,
):
    # If ensembling, we need to make the individual predictions the batch dimension first.
    if inference_cfg["model"]["ensemble"]:
        show_dict = {
            "x": output_dict["x"],
            "y_true": output_dict["y_true"],
            "y_pred": einops.rearrange(output_dict["y_pred"], "1 C E H W -> E C H W"),
        }
    else:
        show_dict = output_dict
    # Show the individual predictions.
    ShowPredictionsCallback(
        show_dict, 
        softpred_dim=1,
        from_logits=inference_cfg["model"]["ensemble"] # If ensemble, need to do a softmax. over ensemble members.
        )
    # If we are showing examples with an ensemble, then we
    # returned initially the individual predictions.
    if inference_cfg["model"]["ensemble"]:
        # Combine the outputs of the models.
        ensemble_prob_map = get_combine_fn(inference_cfg["model"]["ensemble_combine_fn"])(
            output_dict["y_pred"], 
            pre_softmax=inference_cfg["model"]["ensemble_pre_softmax"]
            )
        # Get the hard prediction and probabilities, if we are doing identity,
        # then we don't want to return probs.
        ensemble_prob_map, ensemble_pred_map = process_pred_map(
            ensemble_prob_map, 
            multi_class=True, 
            threshold=0.5,
            from_logits=False, # Ensemble methods already return probs.
            )
        # Place the ensemble predictions in the output dict.
        ensembled_output_dict = {
            "x": output_dict["x"],
            "y_true": output_dict["y_true"],
            "y_pred": ensemble_prob_map,
            "y_hard": ensemble_pred_map
        }
        # Finally, show the ensemble combination.
        ShowPredictionsCallback(
            ensembled_output_dict, 
            softpred_dim=1,
            from_logits=False
            )