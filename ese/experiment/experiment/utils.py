import torch
import einops


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