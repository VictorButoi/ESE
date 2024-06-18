# local imports
from ..metrics.utils import agg_neighbors_preds
# misc imports
import numpy as np
from typing import Literal
from dataclasses import dataclass
from pydantic import validate_arguments
from scipy.ndimage import convolve, gaussian_filter
from albumentations.core.transforms_interface import DualTransform


@validate_arguments
@dataclass(eq=False, repr=False)
class SVLS(DualTransform):

    ksize: int
    sigma: float 
    always_apply: bool
    include_center: bool
    ktype: Literal['gaussian', 'uniform'] = 'gaussian'
    p: float = 0.5

    def __post_init__(self):
        super().__init__(self.always_apply, self.p)
        assert self.ksize % 2 == 1, "Kernel size must be odd"
        self.local_kernel = self.init_filter()

    def apply(self, img, **params):
        # No changes to the image, return as is
        return img

    def apply_to_mask(self, mask, **params):
        # Apply the local kernel to the mask with a convolution
        return convolve(mask, weights=self.local_kernel)


    def init_filter(self):
        # Make the base array that we will normalize.
        if self.ktype == 'gaussian':
            # Create an empty array with the desired size
            filter_array = np.zeros((self.ksize, self.ksize))
            # Place a single 1 in the middle
            filter_array[self.ksize // 2, self.ksize // 2] = 1
            # Apply the Gaussian filter
            kernel_array = gaussian_filter(filter_array, sigma=self.sigma)
        elif self.ktype == 'uniform':
            kernel_array = np.ones((self.ksize, self.ksize))
        else:
            raise ValueError(f"Invalid kernel type: {self.ktype}")

        # If we don't include the center then we zero it out.
        if not self.include_center:
            kernel_array[self.ksize // 2, self.ksize // 2] = 0
        
        # Normalize the kernel to sum to 1.
        normalized_local_kernel = kernel_array / kernel_array.sum()

        return normalized_local_kernel