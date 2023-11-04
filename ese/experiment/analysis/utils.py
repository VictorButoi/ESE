import pandas as pd
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
from ionpy.experiment.util import absolute_import


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dataloader_from_exp(
        exp, 
        new_dset_options=None, 
        return_data_id=False,
        num_workers=1
        ):
    exp_data_cfg = exp.config['data'].to_dict()
    if new_dset_options is not None:
        for key in new_dset_options.keys():
            exp_data_cfg[key] = new_dset_options[key]
    # Make sure we aren't sampling for evaluation. 
    if "slicing" in exp_data_cfg.keys():
        assert exp_data_cfg["slicing"] not in ["central", "dense", "uniform"], "Sampling methods not allowed for evaluation."
    # Get the dataset class and build the transforms
    dataset_cls = exp_data_cfg.pop("_class")
    dataset_obj = absolute_import(dataset_cls)(**exp_data_cfg)
    exp_data_cfg["_class"] = dataset_cls        
    dataset_obj.return_data_id = return_data_id 
    # Build the dataset and dataloader.
    dataloader = DataLoader(
        dataset_obj, 
        batch_size=1, 
        num_workers=num_workers,
        shuffle=False
    )
    return dataloader, exp_data_cfg


def reorder_splits(df):
    train_logs = df[df['split'] == 'train']
    val_logs = df[df['split'] == 'val']
    cal_logs = df[df['split'] == 'cal']
    fixed_df = pd.concat([train_logs, val_logs, cal_logs])
    return fixed_df
