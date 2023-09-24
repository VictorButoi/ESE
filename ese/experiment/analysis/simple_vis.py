# random imports
from typing import Any
from pydantic import validate_arguments

#ese imports
from ese.experiment.augmentation import smooth_soft_pred

#ionpy imports
from ionpy.metrics.segmentation import dice_score

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subject_image(
    subj_name: str,
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    im = ax.imshow(subj["image"], cmap="gray")
    ax.set_title(f"{subj_name}, Image")
    fig.colorbar(im, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subj_label(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    lab = ax.imshow(subj["label"], cmap="gray")
    ax.set_title("Ground Truth")
    fig.colorbar(lab, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_prediction_probs(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    pre = ax.imshow(subj["soft_pred"], cmap="gray")
    ax.set_title("Probabilities")
    fig.colorbar(pre, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_pred(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    # Plot the pixel-wise prediction
    hard_im = ax.imshow(subj["hard_pred"], cmap="gray")

    # Expand extra dimensions for metrics
    pred = subj["hard_pred"][None, None, ...]
    label = subj["label"][None, None, ...]

    # Calculate the dice score
    dice = dice_score(pred, label)
    ax.set_title(f"Prediction, Dice: {dice:.3f}")

    fig.colorbar(hard_im, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_smoothed_pred(
    subj: dict,
    num_bins: int,
    fig: Any = None,
    ax: Any = None
):
    # Plot the processed region-wise prediction
    smoothed_prediction = smooth_soft_pred(subj["soft_pred"], num_bins)
    hard_smoothed_prediction = (smoothed_prediction > 0.5).float()
    smooth_im = ax.imshow(hard_smoothed_prediction, cmap="gray")

    # Expand extra dimensions for metrics
    smoothed_pred = hard_smoothed_prediction[None, None, ...]
    label = subj["label"][None, None, ...]

    # Calculate the dice score
    smoothed_dice = dice_score(smoothed_pred, label)
    ax.set_title(f"Smoothed Prediction, Dice: {smoothed_dice:.3f}")

    fig.colorbar(smooth_im, ax=ax)