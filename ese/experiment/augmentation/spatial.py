import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


class SVLS(DualTransform):
    def __init__(self, ksize, alpha, always_apply=False, p=0.5):
        super(SVLS, self).__init__(always_apply, p)
        self.kernel_size = ksize 
        self.alpha = alpha 

    def apply(self, img, **params):
        # No changes to the image, return as is
        return img

    def apply_to_mask(self, mask, **params):
        return mask

    def local_smooth(self, mask, kernel_size, alpha):
        assert len(mask.shape) == 2, 'Only 2D segmentation masks are supported at the moment.'
        return None