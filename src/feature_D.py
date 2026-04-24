import numpy as np
import cv2

def get_diameter_feature(mask):
    """
    Computes a resolution-normalized diameter feature from a binary mask.

    The diameter is defined as the maximum thickness of the object
    (horizontal or vertical), normalized by the image size so that
    images with different resolutions are comparable.

    Parameters
    ----------
    mask : np.ndarray
        2D array representing a segmentation mask

    Returns
    -------
    float
        Normalized diameter in the range [0, 1], or np.nan if empty
    """

    # Ensure binary mask
    mask = (mask > 0)

    # Check if mask is empty
    if not mask.any():
        return np.nan

    # Image dimensions
    img_height, img_width = mask.shape

    # Height (max pixels in column)
    pixels_in_col = np.sum(mask > 0, axis=0)
    height = np.max(pixels_in_col)

    if height == 0:
        return np.nan

    # Width (max pixels in row)
    pixels_in_row = np.sum(mask > 0, axis=1)
    width = np.max(pixels_in_row)

    if width == 0:
        return np.nan

    # Diameter in pixels
    diameter_px = max(height, width)

    # Normalize by image size (scale-invariant)
    diameter_normalized = diameter_px / max(img_height, img_width)

    return diameter_normalized