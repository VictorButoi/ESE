import pickle


def get_loaded_diseg_results(ResultsLoader, experiment_name, temp_mod=None, root="/storage/vbutoi/scratch"):
    path = f"{root}/{experiment_name}"

    dfc = ResultsLoader.load_configs(
        path,
        properties=False,
    )
    df = ResultsLoader.load_metrics(dfc)

    def model_type(model):
        return model.split('.')[-1]

    def temperature(temperature, init_temp, min_temp):
        if isinstance(temperature, str):
            return f"flexible, {init_temp} ~ {min_temp}"
        else:
            return temperature

    def phase(phase):
        if phase == "train":
            return "val"
        else:
            return "test"

    df.augment(model_type)
    if temp_mod:
        df.augment(temperature)
    df.augment(phase)

    return df


def load_exp_predictions(task, exp_name, root="/storage/vbutoi/scratch/DisegStuff"):
    exp_dir = f"{root}/{task}_{exp_name}_predictions.pickle"
    # Load the distance metrics
    with open(exp_dir, 'rb') as f:
        loaded_dist_df = pickle.load(f)
    return loaded_dist_df