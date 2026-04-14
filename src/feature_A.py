# Extract feature_A: Asymmetry

import numpy as np
from math import floor, ceil
from skimage.transform import rotate
from skimage.measure import label, regionprops


def crop_to_bbox(mask):
    """Crop binary mask to its bounding box using regionprops."""
    labeled = label(mask)
    props = regionprops(labeled)[0]
    minr, minc, maxr, maxc = props.bbox
    return mask[minr:maxr, minc:maxc]

def pad_to_match(a, b):
    """
    Pad the smaller of two 2D arrays with zeros to match the dimensions of the larger.
    Returns (a_padded, b_padded) with identical shape.
    """
    h1, w1 = a.shape
    h2, w2 = b.shape
    target_h = max(h1, h2)
    target_w = max(w1, w2)
    
    a_pad = np.zeros((target_h, target_w), dtype=a.dtype)
    b_pad = np.zeros((target_h, target_w), dtype=b.dtype)
    a_pad[:h1, :w1] = a
    b_pad[:h2, :w2] = b
    return a_pad, b_pad

def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0

def dice(a, b):
    """
    Dice coefficient between two binary arrays.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Ignores background (0,0) pairs.
    """
    intersection = np.logical_and(a, b).sum()
    return 2.0 * intersection / (a.sum() + b.sum() + 1e-8)  # small epsilon to avoid division by zero

def lesion_symmetry(mask, angles=np.arange(0, 180, 2)):
    """
    Compute best symmetry score (0 to 1) for a binary skin lesion mask.
    Higher score = more symmetric.
    Uses Dice coefficient and rectangle (tight bounding box).
    """
    mask = mask.astype(bool)
    if not np.any(mask):
        return 0.0

    # Crop to tight bounding box (rectangle)
    cropped = crop_to_bbox(mask)
    h, w = cropped.shape
    center = (h / 2.0, w / 2.0)  # can be half-integer

    best_sym = 0.0

    for angle in angles:
        # Rotate around the rectangle's center
        rotated = rotate(cropped, angle, center=center, order=0, preserve_range=True).astype(bool)
        h_rot, w_rot = rotated.shape
        
        # Split into left/right and top/bottom at the centre (may be half-integer)
        mid_h = h_rot // 2
        mid_w = w_rot // 2
        
        # Left vs right halves (vertical symmetry)
        left = rotated[:, :mid_w]
        right = rotated[:, mid_w:]
        # Flip right horizontally for comparison
        right_flipped = np.fliplr(right)
        # Pad to same size if necessary
        left_pad, right_pad = pad_to_match(left, right_flipped)
        vert_score = dice(left_pad, right_pad)
        
        # Top vs bottom halves (horizontal symmetry)
        top = rotated[:mid_h, :]
        bottom = rotated[mid_h:, :]
        bottom_flipped = np.flipud(bottom)
        top_pad, bottom_pad = pad_to_match(top, bottom_flipped)
        horz_score = dice(top_pad, bottom_pad)
        
        # Overall symmetry for this angle
        sym = (vert_score + horz_score) / 2.0
        if sym > best_sym:
            best_sym = sym

    return best_sym