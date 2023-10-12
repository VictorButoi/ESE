from build_cityscapes import thunderify_cityscapes
import pathlib


if __name__=="__main__":
    root = pathlib.Path("/storage/vbutoi/datasets/CityScapes/processed")
    dst = pathlib.Path("/storage/vbutoi/datasets/CityScapes/thunder_cityscapes")
    version = '0.2'
    thunderify_cityscapes(root, dst, version)