import torch


def get_calibrator_cls(calibrator: str):
    # Get the calibrator name
    calibrator_class_name_map = {
        "TempScaling": "ese.experiment.models.calibrators.Temperature_Scaling",
        "VectorScaling": "ese.experiment.models.calibrators.Vector_Scaling",
        "DirichletScaling": "ese.experiment.models.calibrators.Dirichlet_Scaling",
        "ImageBasedTS": "ese.experiment.models.calibrators.ImageBasedTS",
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
    

def create_gaussian_tensor(mu, sigma, ksize):
    # Create a grid of (x, y) coordinates
    x = torch.arange(ksize).float()
    y = torch.arange(ksize).float()
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    
    # Compute the center of the grid
    center = (ksize - 1) / 2.0
    
    # Compute the distance of each point from the center
    x_dist = x_grid - center
    y_dist = y_grid - center
    dist_squared = x_dist**2 + y_dist**2
    
    # Compute the Gaussian function
    gaussian_tensor = torch.exp(-dist_squared / (2 * sigma**2))
    
    # Normalize the tensor so that the sum of all elements is 1
    gaussian_tensor /= gaussian_tensor.sum()

    # Shift the mean by mu
    gaussian_tensor += mu
    
    return gaussian_tensor