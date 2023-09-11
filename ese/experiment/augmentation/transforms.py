# ionpy imports
from ionpy.experiment.util import absolute_import

# torch imports
from torchvision.transforms import functional as F
from torchvision import transforms as T


def build_transforms(config):
    """Builds the transforms from the config."""
    # Get the transforms we want to apply
    transforms = []

    for transform_dict in config:
        transform = next(iter(transform_dict.values()))
        transform_cls = absolute_import(transform.pop("_class"))
        transforms.append(transform_cls(**transform))

    return transforms


class RandomCropSegmentation(T.RandomCrop):
    """Randomly crop both image and segmentation mask with the same parameters."""
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        # Ensure same seed for image and mask for consistent cropping
        i, j, h, w = self.get_params(img, self.size)
        
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        
        return {'image': img, 'mask': mask}