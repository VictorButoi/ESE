from build_oxfordpets import proc_OxfordPets 
import pathlib


if __name__=="__main__":
    root = pathlib.Path("/storage/vbutoi/datasets/OxfordPets")

    proc_OxfordPets(
        root, 
        version="0.1",
        show=False,
        save=True
    )
