
def get_calibrator_cls(calibrator: str):
    # Get the calibrator name
    calibrator_class_name_map = {
        "TempScaling": "ese.experiment.models.calibrators.Temperature_Scaling",
        "VectorScaling": "ese.experiment.models.calibrators.Vector_Scaling",
        "DirichletScaling": "ese.experiment.models.calibrators.Dirichlet_Scaling",
        "LTS": "ese.experiment.models.calibrators.LTS",
        "NectarScaling": "ese.experiment.models.calibrators.NECTAR_Scaling",
        "HistogramBinning": "ese.experiment.models.binning.Histogram_Binning",
        "NectarBinning": "ese.experiment.models.binning.NECTAR_Binning",
    }
    if calibrator in calibrator_class_name_map:
        return calibrator_class_name_map[calibrator]
    else:
        return calibrator