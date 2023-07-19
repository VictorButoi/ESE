# IonPy imports
from IonPy.analysis import ResultsLoader
from IonPy.metrics.segmentation import dice_score
from IonPy.util import autoload

# UniverSeg imports
from universeg.experiment.experiment import (BaselineExperiment, FewshotBaselineExperiment,
                          UniversegExperiment)
from universeg.experiment.results import load_configs

# diseg imports
from diseg.experiment.datasets import Segment2D
from diseg.experiment.experiment import DisegExperiment
from diseg.experiment.models.sampler import BaseSampler

# torch imports
import torch
import torch.nn as nn

# misc imports
import numpy as np
import pandas as pd
import pathlib
from pydantic import validate_arguments
from tqdm.notebook import tqdm
from typing import Literal


@validate_arguments
def load_ic_model(
    support_size: int = 8,
    to_gpu: bool = True,
    checkpoint: str = "max-val_od-dice_score"
    ):
    configs = load_configs()
    small_support_model = configs.select(support_size=support_size)
    experiment = load_experiment(small_support_model.path.values[0])
    # Load the model and send to gpu
    experiment.load(tag=checkpoint)
    if to_gpu:
        experiment.to_device()
    # Extract the model and put it in eval mode
    frozen_model = experiment.model
    frozen_model.eval()  # ensure model is in eval mode
    return frozen_model


def load_diseg_model(
    model_path: pathlib.Path = None,
    exp_name: pathlib.Path = None,
    result_loader: ResultsLoader = None,
    to_gpu: bool = True,
    checkpoint_split: str = "train",
    checkpoint: str = "last",
    exp_root: str = "/storage/vbutoi/scratch"
    ):
    assert (model_path is not None) ^ (exp_name is not None), "Either define the path or the directory."
    
    if model_path is not None:
        # Load the model and put it on the gpu
        exp_model_path = model_path
    else:
        assert result_loader is not None, "If model_path is not provided, then result_loader must be provided."
        # pick the best model from the directory loaded
        model_dir = f"{exp_root}/{exp_name}"
        dfc = result_loader.load_configs(
            model_dir,
            properties=False,
        )
        df = result_loader.load_metrics(dfc)
        chosen_df = df.loc[df.select(phase=checkpoint_split)['dice_score'].idxmax()]
        exp_model_path = chosen_df['path']

    # Load the experiment and the correct checkpoint
    experiment = load_experiment(exp_model_path) 
    experiment.load(tag=checkpoint)
    if to_gpu:
        experiment.to_device()
    # Get the selector models
    diseg_model = experiment.model
    diseg_model.eval()

    if model_path is None:
        return diseg_model, chosen_df
    else:
        return diseg_model


@validate_arguments
def load_experiment(path: pathlib.Path,):
    properties = autoload(path / "properties.json")
    exp_name = properties["experiment"]["class"]
    exp_cls = {
        "DisegExperiment": DisegExperiment,
        "UniversegExperiment": UniversegExperiment,
        "FewshotBaselineExperiment": FewshotBaselineExperiment,
        "BaselineExperiment": BaselineExperiment,
    }[exp_name]

    exp = exp_cls(path)
    return exp


def dice_np(y_pred, y_true, from_logits=True):
        score = dice_score(
            y_pred,
            y_true,
            from_logits=from_logits,
            reduction=None,
            batch_reduction=None,
        )
        return score.detach().cpu().numpy().squeeze(1)


#@validate_arguments
@torch.no_grad()
def get_distance_baseline_perf(
    experiment: pathlib.Path,
    task: str,
    split: Literal["val", "test"],
    support_size: int,
    distance_df: pd.DataFrame,
    target_dataset: Segment2D,
    source_dataset: Segment2D,
    comp_pair: Literal["xx", "yy"],
    dist_metric: Literal["mse", "soft_dice", "hard_dice"],
    n_predictions: int = 1,
    top_k: int = -1,
    sample: bool = False,
    replace: bool = False,
    exp_scale: bool = False,
    probabilistic: bool = False,
    checkpoint: str = "max-val_od-dice_score"):

    if isinstance(experiment, (str, pathlib.Path)):
        experiment = load_experiment(experiment)

    experiment.load(tag=checkpoint)
    experiment.to_device()
    model = experiment.model
    model.eval()  # ensure model is in eval mode

    split_dist_df = distance_df.select(split=split,
                                       comp_pair=comp_pair,
                                       dist_metric=dist_metric)
    rows = []

    for i, (x, y) in tqdm(enumerate(target_dataset), desc="Predicting", unit="subject", disable=True, total=len(target_dataset)):
        preds = []

        # Select for particular query subjects
        subj_df = split_dist_df.select(query=i)

        for j in range(n_predictions):
            (sx, sy), indices = dist_based_support(source_dataset=source_dataset,
                                                    support_size=support_size,
                                                    dists=subj_df['metric'].values,
                                                    top_k=top_k,
                                                    sample=sample,
                                                    replace=replace,
                                                    exp_scale=exp_scale,
                                                    probabilistic=probabilistic)

            # the support set
            yhat = model(sx[None], sy[None], x[None])
            preds.append(yhat)
            score = dice_np(yhat, y[None])[0]

            rows.append(
                {
                    "subject": i,
                    "prediction": j,
                    "dice_score": score,
                    "support_indices": tuple(indices),
                    "task": task,
                    "split": split,
                    "support_size": support_size,
                    "loss_func": dist_metric,
                    "comp_pair": comp_pair,
                    "top_k": top_k,
                    "sample": sample,
                    "replace": replace,
                    "exp_scale": exp_scale,
                    "probabilistic": probabilistic,
                }
            )

    return pd.DataFrame.from_records(rows)

    
def dist_based_support(source_dataset: Segment2D,
                        support_size: int,
                        dists: np.ndarray,
                        top_k: int,
                        sample: bool,
                        exp_scale: bool,
                        replace: bool,
                        probabilistic: bool):
    assert not (replace and not sample), "If replace is True, then sample must be True."
    assert not (exp_scale and not sample), "If exp_scale is True, then sample must be True."

    def k_smallest_indices(arr, k):
        indices = np.argpartition(arr, k)[:k]
        return indices

    def sample_from_dist(dists, support_size, replace, exp_scale):
        # Convert distance array to torch tensor
        dists = torch.from_numpy(dists)
        
        # Invert to make closer distances have higher probability
        closesness = -dists

        # Exp scale for unbounded distances
        if exp_scale:
            closesness = torch.exp(closesness)

        # Calculate how the probability of each element should be scaled
        if probabilistic:
            closesness = torch.softmax(closesness, dim=0)

        # Sample from the distribution to choose the support set
        return torch.multinomial(closesness, support_size, replacement=replace)

    # if top k is not None, then we sample from the top k SMALLEST distances.
    if top_k != -1:
        smallest_indices = k_smallest_indices(dists, k=top_k)
        if sample:
            assert not(support_size > top_k and not replace), "If support size is larger than top k, then replace must be True."
            dist_copy = dists.copy()  # set all elements bigger than the smallest indices to 0
            dist_copy[dist_copy > dist_copy[smallest_indices[-1]]] = 0
            idxs = sample_from_dist(dist_copy, support_size, replace, exp_scale)
        else:
            assert top_k == support_size, "If no sampling, k must be equal to support size."
            assert not probabilistic, "If no sampling, probabilistic must be False."
            idxs = smallest_indices
    else:
        assert not(top_k is None and not sample), "If top_k is None, then sampling must be True."
        idxs = sample_from_dist(dists, support_size, replace, exp_scale)

    imgs, segs = zip(*(source_dataset[i] for i in idxs))
    
    return (torch.stack(imgs), torch.stack(segs)), tuple(idxs.tolist())


#@validate_arguments
@torch.no_grad()
def compute_diseg_breakdown(
    selector_model: nn.Module,
    frozen_model: nn.Module,
    split_datasets: dict[str, Segment2D],
    run_args: dict,
    task: str,
    split: Literal["val", "test"],
    sampler: Literal['gumbel', 'multinomial'],
    replace: bool,
    n_predictions: int = 1,
    disable_tqdm: bool = True):

    assert not(not replace and sampler=="gumbel"), "If sampler is gumbel, then replace must be True."
    run_args_copy = run_args.copy()

    # Choose the target dataset
    source_dataset = split_datasets["train"]
    target_dataset = split_datasets[split]

    # Sample your supports either by gumbel or multinomial
    if sampler == "gumbel":
        selection_module = selector_model.sampler
        run_args_copy["sampler"] = "GumbelSampler"
    else:
        selection_module = BaseSampler(support_size=selector_model.sampler.support_size,
                                        batch_size=selector_model.sampler.batch_size,
                                        replacement=replace)
        run_args_copy["sampler"] = "BaseSampler"

    # Keep list of records
    rows = []

    for i, (x, y) in tqdm(enumerate(target_dataset), 
                        desc="Predicting", 
                        unit="subject", 
                        disable=disable_tqdm, 
                        total=len(target_dataset)):
        
        # Extend dim for batch 
        x = x[None]
        y = y[None]

        for j in range(n_predictions):
            
            # Choose the indices to be selected and select them
            indices_logits = selector_model(x, y)
            
            # Sample support and get indices
            sx, sy, _ = selection_module(indices_logits, source_dataset)

            # Copy the query image multiple times in the batch dimension
            query_images = x.repeat(selector_model.sampler.batch_size, 1, 1, 1)
            query_labels = y.repeat(selector_model.sampler.batch_size, 1, 1, 1)

            # the support set
            yhat = frozen_model(sx, sy, query_images)
            score = dice_np(yhat, query_labels)[0]

            rows.append(
                {
                    "subject": i,
                    "prediction": j,
                    "dice_score": score,
                    "replace": replace,
                    "task": task,
                    "split": split,
                    **run_args_copy
                }
            )
    
    # Wrap everything with a big dataframe bow
    return pd.DataFrame.from_records(rows)