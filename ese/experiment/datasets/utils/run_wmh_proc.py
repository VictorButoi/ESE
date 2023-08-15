from build_wmh import proc_WMH 
import pathlib

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

if __name__=="__main__":
    root = pathlib.Path("/storage/vbutoi/datasets/WMH")
    splits = ["training", "test", "additional_annotations"]

    all_dirs = []

    for split in splits:
        split_path = root / split
        for subdir in split_path.iterdir():
            print(subdir)
            all_dirs.append(subdir)
            for l3_dir in subdir.iterdir():
                if not is_integer(str(l3_dir.name)):
                    print(l3_dir)
                    all_dirs.append(l3_dir)
                    for l4_dir in l3_dir.iterdir():
                        if not is_integer(str(l4_dir.name)):
                            print(l4_dir)
                            all_dirs.append(l4_dir)
                            for l5_dir in l4_dir.iterdir():
                                if not is_integer(str(l5_dir.name)):
                                    print(l5_dir)
                                    all_dirs.append(l5_dir)
    
    unique_dirs = []
    for path in all_dirs:
        all_other_dirs = [p for p in all_dirs if p != path]
        is_subdir = False
        for other_path in all_other_dirs:
            if path in other_path.parents:
                is_subdir = True
                break
        if not is_subdir:
            unique_dirs.append(path)

    proc_WMH(
        unique_dirs, 
        modalities=["FLAIR"],
        show=False,
        save=True
    )