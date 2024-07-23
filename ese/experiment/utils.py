# torch imports
import os
import torch
# ionpy imports
from ionpy.analysis import ResultsLoader
from ionpy.experiment.util import absolute_import
# misc imports
import ast
import json
import einops
from pathlib import Path
import matplotlib.pyplot as plt
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
    from_logits: bool,
    threshold: float = 0.5, 
    ):
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
    inference_cfg: dict,
    ens_weights: Optional[torch.Tensor] = None,
) -> dict:
    if "ens_weights" in output_dict:
        ens_weights = output_dict["ens_weights"]
    # Get a few variables from the ensemble config.
    combine_fn = inference_cfg['ensemble']['combine_fn']
    norm_ensemble = inference_cfg['ensemble']['normalize']
    combine_quantity = inference_cfg['ensemble']['combine_quantity']
    # If the probs are provided, then we don't need to convert the logits to probs.
    if combine_quantity == "logits":
        # Combine the outputs of the models.
        ensemble_prob_map = get_combine_fn(combine_fn)(
            output_dict["y_logits"], 
            combine_quantity="logits",
            weights=ens_weights,
            normalize=norm_ensemble,
            from_logits=True
        )
    else:
        prob_args = {
            "combine_quantity": "probs",
            "weights": ens_weights,
            "normalize": norm_ensemble,
        }
        if output_dict["y_probs"] is not None:
            # Combine the outputs of the models.
            ensemble_prob_map = get_combine_fn(combine_fn)(
                output_dict["y_probs"], 
                from_logits=False,
                **prob_args
            )
        else:
            assert output_dict["y_logits"] is not None, "No logits or probs provided."
            # Combine the outputs of the models.
            ensemble_prob_map = get_combine_fn(combine_fn)(
                output_dict["y_logits"], 
                from_logits=True,
                **prob_args
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
        "y_probs": ensemble_prob_map, # (B, C, H, W)
        "y_hard": ensemble_pred_map # (B, C, H, W)
    }

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_experiment(
    checkpoint: str,
    device: str = "cuda",
    set_seed: bool = True,
    load_data: bool = True,
    df: Optional[Any] = None, 
    path: Optional[str] = None,
    attr_dict: Optional[dict] = None,
    exp_class: Optional[str] = None,
    selection_metric: Optional[str] = None,
):
    if path is None:
        assert df is not None, "Must provide a dataframe if no path is provided."
        if attr_dict is not None:
            for attr_key in attr_dict:
                select_arg = {attr_key: attr_dict[attr_key]}
                if attr_key in ["mix_filters"]:
                    select_arg = {attr_key: ast.literal_eval(attr_dict[attr_key])}
                df = df.select(**select_arg)
        if selection_metric is not None:
            phase, score = selection_metric.split("-")
            df = df.select(phase=phase)
            df = df.sort_values(score, ascending=False)
        exp_path = df.iloc[0].path
    else:
        assert attr_dict is None, "Cannot provide both a path and an attribute dictionary."
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
    exp_obj = exp_class(
        exp_path, 
        init_metrics=False, 
        load_data=load_data,
        set_seed=set_seed
    )

    # Load the experiment
    if checkpoint is not None:
        # Very scuffed, but sometimes we want to load different checkpoints.
        try:
            print(f"Loading checkpoint: {checkpoint}.")
            exp_obj.load(tag=checkpoint)
        except Exception as e:
            print(e)
            print("Defaulting to loading: max-val-dice_score.")
            exp_obj.load(tag="max-val-dice_score") # Basically always have this as a checkpoint.
    
    # Set the device
    exp_obj.device = torch.device(device)
    if device == "cuda":
        exp_obj.to_device()
    
    return exp_obj


def get_exp_load_info(pretrained_exp_root):
    is_exp_group = not ("config.yml" in os.listdir(pretrained_exp_root)) 
    # Load the results loader
    rs = ResultsLoader()
    # If the experiment is a group, then load the configs and build the experiment.
    if is_exp_group: 
        dfc = rs.load_configs(
            pretrained_exp_root,
            properties=False,
        )
        return {
            "df": rs.load_metrics(dfc),
        }
    else:
        return {
            "path": pretrained_exp_root
        }


def show_inference_examples(
    output_dict,
    inference_cfg,
):
    hard_pred_threshold = inference_cfg['experiment']['hard_pred_threshold']

    # If ensembling, we need to make the individual predictions the batch dimension first.
    if inference_cfg["model"].get("ensemble", False):
        show_dict = {
            "x": output_dict["x"],
            "y_true": output_dict["y_true"],
        }
        if "y_probs" in output_dict and output_dict["y_probs"] is not None:
            show_dict["y_probs"] = einops.rearrange(output_dict["y_probs"], "1 C E H W -> E C H W")
            show_first_pred = False if show_dict["y_probs"].shape[0] == 1 else True
        elif "y_logits" in output_dict and output_dict["y_logits"] is not None:
            show_dict["y_logits"] = einops.rearrange(output_dict["y_logits"], "1 C E H W -> E C H W"),
            show_first_pred = False if show_dict["logits"].shape[0] == 1 else True
    else:
        show_first_pred = True
        show_dict = output_dict
    # Show the individual predictions of the ensemble, or just the prediction if no ensemble.
    if show_first_pred:
        ShowPredictionsCallback(show_dict, threshold=hard_pred_threshold)
    # If we are showing examples with an ensemble, then we show initially the individual predictions.
    if inference_cfg["model"].get("ensemble", False):
        # Combine the outputs of the models.
        ensemble_outputs = reduce_ensemble_preds(
            output_dict=output_dict, 
            inference_cfg=inference_cfg
        )
        # Place the ensemble predictions in the output dict.
        ensembled_output_dict = {
            "x": output_dict["x"],
            "y_true": output_dict["y_true"],
            "y_probs": ensemble_outputs["y_probs"],
            "y_hard": ensemble_outputs["y_hard"] 
        }
        # Finally, show the ensemble combination.
        ShowPredictionsCallback(ensembled_output_dict, threshold=hard_pred_threshold)
    # Show the support examples if they are provided.
    if "support_set" in output_dict:
        support_images = output_dict["support_set"]["context_images"]
        support_labels = output_dict["support_set"]["context_labels"]
        support_size = support_images.shape[1]
        assert support_images.shape[0] == 1, "Support set must have a batch size of 1."
        f, axarr = plt.subplots(2, support_size, figsize=(support_size * 3, 6))
        for supp_idx in range(support_size):
            axarr[0, supp_idx].imshow(support_images[:, supp_idx, ...].squeeze().cpu().numpy(), cmap="gray", interpolation="none")
            axarr[1, supp_idx].imshow(support_labels[:, supp_idx, ...].squeeze().cpu().numpy(), cmap="gray", interpolation="none")
            axarr[0, supp_idx].axis("off")
            axarr[1, supp_idx].axis("off")
        plt.show()