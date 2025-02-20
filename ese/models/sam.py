# SAM imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# torch imports
from torch import nn
from torch import Tensor
# misc imports
from typing import Optional
from dataclasses import dataclass
from pydantic import validate_arguments


@validate_arguments
@dataclass(eq=False, repr=False)
class SAM(nn.Module):

    model_cfg: str
    checkpoint: str
    num_classes: int
    freeze_encoder: bool = True
    out_activation: Optional[str] = None

    def __post_init__(self):
        super().__init__()

        sam2_model = build_sam2(self.model_cfg, self.checkpoint, device="cuda") # load model
        predictor = SAM2ImagePredictor(sam2_model)

        # Set training parameters
        predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
        predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

    def reset_parameters(self):
        """Reset parameters for the mask decoder's task-specific head."""
        for module in self.sam.mask_decoder.task_head.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, 
        x: Tensor, 
        input_point: Tensor,
        input_label: Tensor
    ) -> Tensor:
        # Set the image for the predictor
        self.predictor.set_image(x) # apply SAM image encoder to the image

        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
            input_point, 
            input_label, 
            box=None, 
            mask_logits=None, 
            normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=None,
            masks=None
        )

        # mask decoder
        batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features
        )
        prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])# Upscale the masks to the original image resolution

        return prd_masks 

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device