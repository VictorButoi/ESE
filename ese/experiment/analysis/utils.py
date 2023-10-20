import numpy as np
from scipy.ndimage import convolve


def count_matching_neighbors(img):
    # Define a kernel that sums up all 8 neighbors
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # Convolve the image with the kernel. This will give a sum of all 8 neighbors for each pixel.
    neighbor_sums = convolve(np.ones_like(img), kernel, mode='constant', cval=0)

    # Create a mask where the original image matches its neighbors
    matching_neighbors = (img == np.roll(img, 1, axis=0)) | \
                         (img == np.roll(img, -1, axis=0)) | \
                         (img == np.roll(img, 1, axis=1)) | \
                         (img == np.roll(img, -1, axis=1)) | \
                         (img == np.roll(np.roll(img, 1, axis=0), 1, axis=1)) | \
                         (img == np.roll(np.roll(img, -1, axis=0), 1, axis=1)) | \
                         (img == np.roll(np.roll(img, 1, axis=0), -1, axis=1)) | \
                         (img == np.roll(np.roll(img, -1, axis=0), -1, axis=1))

    # Use the mask to filter out the sums
    return neighbor_sums * matching_neighbors