


def calibration_map(
    pred,
    label,
    threshold=0.5
):
    """
    Compute the calibration map for a single image.
    
    Args:
        pred: The soft predicted probability map.
        label: The groundtruth label map.

    Returns:
        calibration_image: The calibration map for the image.
    """
    hard_pred = (pred > threshold).float()
    pixelwise_accuracy = (hard_pred == label).float()

    # Set the regions of the image corresponding to groundtruth label.
    calibration_image = (foreground_accuracy - soft_foreground_pred)

    return calibration_image
