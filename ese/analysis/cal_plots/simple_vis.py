# random imports
from typing import Any
from pydantic import validate_arguments

#ionpy imports
from ionpy.metrics.segmentation import dice_score

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subject_image(
    subj_name: str,
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    image = subj["image"]
    # If rgb, move channels to the back.
    if len(image.shape) == 3:
        image = image.permute(1, 2, 0).int()
    # Plot the image
    im = ax.imshow(image, cmap="gray")
    ax.set_title(f"{subj_name}, Image")
    fig.colorbar(im, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subj_label(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    lab = ax.imshow(subj["label"], interpolation="None", cmap="gray")
    ax.set_title("Ground Truth")
    fig.colorbar(lab, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_conf_map(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    pre = ax.imshow(subj["conf_map"], interpolation="None", cmap="gray")
    ax.set_title("Probabilities")
    fig.colorbar(pre, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_pred(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    # Plot the pixel-wise prediction
    hard_im = ax.imshow(subj["pred_map"], interpolation="None", cmap="gray")

    # Expand extra dimensions for metrics
    pred = subj["pred_map"][None, None, ...]
    label = subj["label"][None, None, ...]

    # Calculate the dice score
    dice = dice_score(pred, label)
    ax.set_title(f"Prediction, Dice: {dice:.3f}")

    fig.colorbar(hard_im, ax=ax)