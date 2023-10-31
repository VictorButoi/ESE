import numpy as np
from scipy.signal import convolve2d

def count_matching_neighbors(label_map):
    # Get unique labels
    unique_labels = np.unique(label_map)

    # Create an array to store the counts
    count_array = np.zeros_like(label_map)

    # Define a 3x3 kernel of ones
    kernel = np.ones((3, 3))
    for label in unique_labels:
        # Create a binary mask for the current label
        mask = (label_map == label).astype(float)
        # Convolve the mask with the kernel to get the neighbor count
        neighbor_count = convolve2d(mask, kernel, mode='same')
        # Update the count_array where the label_map matches the current label
        count_array[label_map == label] = neighbor_count[label_map == label]

    # Subtract 1 because the center pixel is included in the 3x3 neighborhood count
    count_array -= 1
    return count_array