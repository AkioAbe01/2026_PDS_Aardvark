import numpy as np
import cv2

def get_diameter_feature(mask):
    # Ensure binary
    mask = (mask > 0).astype('uint8') * 255

    # Check if mask is empty
    if np.sum(mask) == 0:
        return np.nan

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

    # Return diameter
    diameter = max(height, width)
    return diameter