
def get_calibrator_cls(calibrator: str):
    # Get the calibrator name
    calibrator_class_name_map = {
        "TempScaling": "ese.experiment.models.calibrators.Temperature_Scaling",
        "VectorScaling": "ese.experiment.models.calibrators.Vector_Scaling",
        "DirichletScaling": "ese.experiment.models.calibrators.Dirichlet_Scaling",
        "IBTS": "ese.experiment.models.calibrators.IBTS",
        "LTS": "ese.experiment.models.calibrators.LTS",
        "LocalTS": "ese.experiment.models.calibrators.LocalTS",
        "HistogramBinning": "ese.experiment.models.binning.Histogram_Binning",
        "ContextualHistogramBinning": "ese.experiment.models.binning.Contextual_Histogram_Binning",
        "LocalHistogramBinning": "ese.experiment.models.binning.Local_Histogram_Binning",
        "NectarBinning": "ese.experiment.models.binning.NECTAR_Binning",
        "NectarScaling": "ese.experiment.models.nectar.NECTAR_Scaling",
        "SoftNectarBinning": "ese.experiment.models.binning.Soft_NECTAR_Binning",
    }
    if calibrator in calibrator_class_name_map:
        return calibrator_class_name_map[calibrator]
    else:
        print(f"WARNING! Calibrator not found, using default calibrator name: {calibrator}.")
        return calibrator