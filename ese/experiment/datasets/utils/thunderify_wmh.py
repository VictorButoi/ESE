from build_wmh import thunderify_wmh
import pathlib


if __name__=="__main__":
    root = pathlib.Path("/storage/vbutoi/datasets/WMH/processed")
    dst = pathlib.Path("/storage/vbutoi/datasets/WMH/thunder_wmh")
    version = '0.2'
    thunderify_wmh(root, dst, version)