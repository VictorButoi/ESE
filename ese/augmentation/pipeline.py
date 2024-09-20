
import torch
import voxynth


def build_aug_pipeline(
    augs_dict
):
    spatial_augs = augs_dict.get('spatial', None)
    visual_augs = augs_dict.get('visual', None)
    if visual_augs is not None:
        use_mask = visual_augs.get('use_mask', False)

    def aug_func(x_batch, y_batch):
        
        # Apply augmentations that affect the spatial properties of the image, by applying warps.
        if spatial_augs is not None:
            trf = voxynth.transform.random_transform(x_batch.shape[2:]) # We avoid the batch and channels dims.
            # Put the trf on the device.
            trf = trf.to(x_batch.device)
            # Apply the spatial deformation to each elemtn of the batch.  
            spat_aug_x = torch.stack([voxynth.transform.spatial_transform(x, trf) for x in x_batch])
            spat_aug_y = torch.stack([voxynth.transform.spatial_transform(y, trf) for y in y_batch])
        else:
            spat_aug_x, spat_aug_y = x_batch, y_batch

        # Apply augmentations that affect the visual properties of the image, but maintain originally
        # ground truth mapping.
        if visual_augs is not None:
            if use_mask:
                # Voxynth methods require that the channel dim is squeezed to apply the intensity augmentations.
                if y_batch.ndim != x_batch.ndim - 1:
                    # Try to squeeze out the channel dimension.
                    y_batch = y_batch.squeeze(1)
                aug_x = torch.stack([voxynth.image_augment(x, y, **visual_augs) for x, y in zip(spat_aug_x, spat_aug_y)])
            else:
                aug_x = torch.stack([voxynth.image_augment(x, **visual_augs) for x in spat_aug_x])
        else:
            aug_x = spat_aug_x
        
        return aug_x, spat_aug_y 
    
    return aug_func