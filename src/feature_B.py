import numpy as np
import cv2

def get_border_feature(mask):
    # Ensure binary
    mask = (mask > 0).astype('uint8') * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.nan

    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    if area == 0:
        return np.nan

    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter < 1e-5:
        return np.nan

    # Compactness
    compactness = (4 * np.pi * area) / (perimeter ** 2)
    return compactness

#Interpretation
# ≈1 for a near-perfect circle
# <1 for irregular shapes