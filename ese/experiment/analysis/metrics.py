# diseg imports
from diseg.experiment.datasets import RandomSupport, Segment2D
from diseg.experiment.datasets.inference import get_dataset
from diseg.experiment.analysis.inference import (get_distance_baseline_perf, compute_diseg_breakdown, 
                                                 load_experiment, load_ic_model, load_diseg_model)
from diseg.experiment.analysis.utils import get_loaded_diseg_results

# torch imports
import torch
from torch import nn

# ionpy imports
from IonPy.analysis import ResultsLoader
from IonPy.datasets import CUDACachedDataset
from IonPy.metrics.segmentation import dice_score

# universeg imports
from universeg.experiment.results import load_configs
from universeg.experiment.augmentation import augmentations_from_config
from universeg.experiment.datasets import MultiBinarySegment2DIndex

# misc imports
import pandas as pd
from pydantic import validate_arguments
import os
import pickle
from tqdm.notebook import tqdm
from typing import Any, Dict, List, Literal, Optional


def compute_distance_df(
        split: str, 
        source_dataset: Segment2D,
        target_dataset: Segment2D, 
        dist_metric_dict: dict
    ) -> pd.DataFrame:

    # Keep track of each pairwise loss
    dist_df_rows = []

    # Iterate over each target subject
    for targ_idx in range(len(target_dataset)):

        # Get the target subject
        target_x, target_y = target_dataset[targ_idx]
        target_x = target_x.squeeze()
        target_y = target_y.squeeze()

        # Iterate over each source subject
        for source_index in range(len(source_dataset)):

            # Get the source subject
            source_x, source_y = source_dataset[source_index]

             # compute distances
            for comp_pair in ["xx", "yy"]:

                # get which pair to compare
                if comp_pair == "xx":
                    comp_target = target_x
                    comp_source = source_x.squeeze()
                else:
                    comp_target = target_y
                    comp_source = source_y.squeeze()

                # Iterate over each distance metric
                for dist_metric_key in dist_metric_dict.keys():

                    # compute distance
                    source_to_target = dist_metric_dict[dist_metric_key](comp_target, comp_source)

                    if dist_metric_key == "dice_score":
                        source_to_target = 1 - source_to_target # Flip this around to match the direction of others.

                    # convert to regular float
                    if type(source_to_target) == torch.Tensor:
                        source_to_target = source_to_target.item()

                    # add these to the dictionary
                    row = {
                        "query": targ_idx, 
                        "support": source_index,
                        "split": split,
                        "comp_pair": comp_pair,
                        "dist_metric": dist_metric_key,
                        "metric": source_to_target
                    }
                    dist_df_rows.append(row)

    return pd.DataFrame(dist_df_rows)


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataset: Segment2D,
    support: RandomSupport,
    context_dset_size: int,
    base_seed: int = 10_000,
    n_predictions: int = 5,
    ensemble: bool = False,
    from_logits: bool = True,
    augmentations: Optional[List[Dict[str, Any]]] = None,
    preload_cuda: bool = True,
) -> pd.DataFrame:

    def dice_np(y_pred, y_true):

        score = dice_score(
            y_pred,
            y_true,
            from_logits=from_logits,
            reduction=None,
            batch_reduction=None,
        )
        return score.detach().cpu().numpy().squeeze(1)

    model.eval()  # ensure model is in eval mode

    rows = []
    if augmentations is not None:
        aug_pipeline = augmentations_from_config(augmentations)

    for i, (x, y) in tqdm(enumerate(dataset), desc="Predicting", unit="subject", total=len(dataset)):
        preds = []
        if not preload_cuda:
            x, y = x.cuda(), y.cuda()

        for j in range(n_predictions):

            # Note: different subjects will use different support sets
            # but different models will use the same support sets
            rng = base_seed * (j + 1) + i
            (sx, sy), indices = support[rng]
           
            # Track which entries are being used in the support sets.
            usage_dict = {k : 0 for k in range(context_dset_size)}
            for idx in indices:
                usage_dict[idx] += 1
            
            if not preload_cuda:
                sx, sy = sx.cuda(), sy.cuda()

            if augmentations is not None:
                sx, sy = aug_pipeline.support_forward(sx[None], sy[None])
                sx, sy = sx[0], sy[0]

            # the support set
            yhat = model(sx[None], sy[None], x[None])

            if isinstance(yhat, dict):
                yhat = torch.argmax(torch.softmax(yhat["pred"], axis=1), axis=1)[
                    :, None, ...
                ].float()

            preds.append(yhat)
            score = dice_np(yhat, y[None])[0]

            rows.append(
                {
                    "subject": i,
                    "prediction": j,
                    "rng": rng,
                    "ensemble": False,
                    "dice_score": score,
                    "support_indices": tuple(indices),
                    **usage_dict
                }
            )

        if ensemble:
            y_ens = torch.stack(preds).mean(dim=0)
            score = dice_np(y_ens, y[None])[0]
            rows.append({"subject": i, 
                         "ensemble": True, 
                         "dice_score": score})

    df = pd.DataFrame.from_records(rows)

    for k, v in dataset.signature.items():
        df[k] = v

    return df


@validate_arguments
def compute_random_predictions(
    support_size: int,
    splits: List[str],
    datasets: List[str],
    tasks: Optional[List[str]] = None,
    support_replacement: bool = True,
    context_split: Literal["train", "same"] = "train",
    slicing: Optional[str] = "midslice",
    ensemble: bool = False,
    n_predictions: int = 5,
    support_seed: int = 10_000,
    checkpoint: str = "max-val_od-dice_score",
    preload_cuda: bool = True,
    gpu: int = 0,
    save: bool = False,
) -> pd.DataFrame:
    
    assert len(tasks) == 1, "Only one task is supported for now."

    # Make gpu available
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    # Get your experiment
    configs = load_configs()
    small_support_model = configs.select(support_size=support_size)
    experiment = small_support_model.path.values[0]
    experiment = load_experiment(experiment)
    experiment.load(tag=checkpoint)
    experiment.to_device()

    cfg = experiment.config

    from_logits: bool = cfg.get("loss_func.from_logits", False)
    slicing = slicing or cfg["data.slicing"]

    index = MultiBinarySegment2DIndex()
   
    task_df = index.task_df(
        slicing=slicing,
        datasets=datasets,
        resolution=cfg.get("data.resolution"),
        version=cfg.get("data.version"),
        expand_labels=True,
    )

    if tasks:
       task_df = task_df[task_df.full_task.isin(tasks)]

    dfs = []

    assert len(task_df) == 1, "Must have task be fixed for now."

    for split in splits:
        for _, row in task_df.iterrows():
            copy_keys = ("task", "label", "resolution", "slicing", "version")
            # Build datasets from splits
            segment2d_params = dict(
                split=split,
                min_label_density=0,
                preload=True,
                **{k: row[k] for k in copy_keys},
            )

            target_dataset = Segment2D(**segment2d_params)

            if context_split == "same":
                context_dataset = target_dataset
            else:
                context_dataset = target_dataset.other_split(context_split)

            target_dataset = CUDACachedDataset(target_dataset)
            if context_dataset is not target_dataset:
                context_dataset = CUDACachedDataset(context_dataset)

            support = RandomSupport(
                context_dataset, support_size, replacement=support_replacement, include_indices=True
            )

            df = predict_dataset(
                experiment.model,
                target_dataset,
                support,
                context_dset_size=len(context_dataset),
                ensemble=ensemble,
                n_predictions=n_predictions,
                from_logits=from_logits,
                augmentations=None,
                base_seed=support_seed,
                preload_cuda=preload_cuda
            )
            df['split'] = split

            dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    full_df.attrs.update(
        {
            "inference_support_size": support_size,
            "context_split": context_split,
            "support_replacement": support_replacement,
            "checkpoint": checkpoint,
            "checkpoint_epoch": experiment._checkpoint_epoch,
            "n_predictions": n_predictions,
            "support_seed": support_seed,
        }
    )

    if save:
        # Config optiions
        root = "/storage/vbutoi/scratch/DisegStuff"
        task_comps = tasks[0].split('/')
        out_file_name = task_comps[1] + "_" + task_comps[2] + "_" + task_comps[4]

        # Save to output file
        full_df.to_pickle(f"{root}/{out_file_name}_random_predictions.pickle")

    return full_df
    

@validate_arguments
def compute_predictions_by_metric(
    configure_list, 
    task_name: str, 
    save: bool=True,
    warning: Literal["ignore", "verbose", "raise"]= "ignore"
    ):

    # Config optiions
    root = "/storage/vbutoi/scratch/DisegStuff"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    task_comps = task_name.split('/')
    out_file_name = task_comps[1] + "_" + task_comps[2] + "_" + task_comps[4]

    # Load the desired model
    configs = load_configs()
    small_support_model = configs.select(support_size=8)
    model = small_support_model.path.values[0]

    # Set cuda to True to load the dataset on the GPU
    tasks = [task_name]
    split_datasets = get_dataset(tasks, cuda=True, splits=('train', 'val', 'test'))

    # Load the distance metrics
    dist_file = f"{root}/{out_file_name}_distances.pickle"
    with open(dist_file, 'rb') as f:
        dist_df = pickle.load(f)
    
    # keep track of the results 
    running_df = pd.DataFrame([])
    for option_product in configure_list:
        # some configurations are invalid, we make sure to catch these ahead of time.
        try:
            target_dset = split_datasets[option_product['split']]
            cfg_n_samples = option_product.pop('n_sample_predictions')
            n_to_pred = cfg_n_samples if option_product['sample'] else 1
            option_df = get_distance_baseline_perf(experiment=model,
                                                    task=out_file_name,
                                                    support_size=8,
                                                    distance_df=dist_df,
                                                    target_dataset=target_dset,
                                                    source_dataset=split_datasets['train'],
                                                    n_predictions=n_to_pred,
                                                    **option_product)
            running_df = pd.concat([running_df, option_df])
        except Exception as e:
            if warning == "ignore":
                continue
            elif warning == "verbose":
                print(e)
            else:
                raise Exception(e)

    if save:
        # Save to output file
        running_df.to_pickle(f"{root}/{out_file_name}_predictions_by_metric.pickle")
    
    return running_df


@validate_arguments
def compute_diseg_prediction_metrics(
    configure_list, 
    exp_name: str, 
    task_name: str, 
    disable_tqdm: bool=True,
    save: bool=True, 
    warning: Literal["ignore", "verbose", "raise"]="ignore",
    gpu: int=0
    ):

    # Config optiions
    root = "/storage/vbutoi/scratch/DisegStuff"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    task_comps = task_name.split('/')
    out_file_name = task_comps[1] + "_" + task_comps[2] + "_" + task_comps[4]

    # Load the metrics from trained diseg models
    exps_df = get_loaded_diseg_results(ResultsLoader(), exp_name)

    # Load frozen universeg model
    frozen_model = load_ic_model()

    # Build datasets
    tasks = [task_name]
    split_datasets = get_dataset(tasks, blockify=True, cuda=True, splits=('train', 'val', 'test'))

    unique_model_paths = exps_df["path"].unique()

    # keep track of the results 
    running_df = pd.DataFrame([])

    for model_path in tqdm(exps_df["path"].unique(),
                            desc="Computing model breakdown",
                            unit="model",
                            disable=disable_tqdm,
                            total=len(unique_model_paths)):
        
        # Get the run's arguments
        run_df = exps_df.select(path=model_path).copy()
        pruned_run_df = run_df.drop(['epoch', 'loss', 'dice_score'], axis=1)
        run_args = pruned_run_df.iloc[0].to_dict()

        # Load the model and put it on the gpu
        selector_model = load_diseg_model(model_path=model_path)
    
        for option_product in configure_list:
            # some configurations are invalid, we make sure to catch these ahead of time.
            try:
                option_df = compute_diseg_breakdown(selector_model=selector_model,
                                                    frozen_model=frozen_model,
                                                    split_datasets=split_datasets,
                                                    run_args=run_args,
                                                    task=out_file_name,
                                                    disable_tqdm=disable_tqdm,
                                                    **option_product)
                running_df = pd.concat([running_df, option_df])
            except Exception as e:
                if warning == "ignore":
                    continue
                elif warning == "verbose":
                    print(e)
                else:
                    raise e
                
    if save:
        # Save to output file
        running_df.to_pickle(f"{root}/{out_file_name}_{exp_name}_predictions.pickle")