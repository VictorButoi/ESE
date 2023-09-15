# Kornia imports
from kornia.constants import Resample, SamplePadding
import kornia.augmentation as KA


def RandomScale(scale, **kwargs):
    return KA.RandomAffine(degrees=0.0, translate=0.0, scale=scale, shear=0.0, **kwargs)


def RandomTranslate(translate, **kwargs):
    return KA.RandomAffine(
        degrees=0.0, translate=translate, scale=0.0, shear=0.0, **kwargs
    )
    

def RandomShear(shear, **kwargs):
    return KA.RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=shear, **kwargs)