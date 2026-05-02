# Extract feature_A: Asymmetry

import numpy as np
from skimage.transform import rotate
import skimage.measure

# =====================================================================

def modified_mask(mask):
    binary_mask = (mask > 0.5).astype(bool)
    labeled = skimage.measure.label(binary_mask)
    props = skimage.measure.regionprops(labeled)
    if not props:
        return binary_mask
    largest_region = max(props, key=lambda r: r.area)
    largest_mask = (labeled == largest_region.label).astype(bool)
    aux_mask = binary_mask.copy()
    aux_mask[largest_mask] = False
    return np.logical_xor(binary_mask, aux_mask)

def crop_to_bbox(mask, img):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.nan
    minr, maxr = ys.min(), ys.max()
    minc, maxc = xs.min(), xs.max()
    return mask[minr:maxr, minc:maxc], img[minr:maxr, minc:maxc, :]

def iou(a, b):
    """Jaccard (IoU) between two binary arrays of possibly different sizes.
    Crops both to the minimum dimensions before computing."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0

def lesion_symmetry(cropped_mask, angles=np.arange(0, 180, 22.5)):
    """mask is already preprocessed, binary, and downsampled."""
    if cropped_mask.sum() < 100:
        return 0.0

    h, w = cropped_mask.shape
    center = (h / 2.0, w / 2.0)
    best_sym = 0.0

    for angle in angles:
        rotated = rotate(cropped_mask, angle, center=center, order=0, preserve_range=True).astype(bool)
        hh, ww = rotated.shape
        mh, mw = hh // 2, ww // 2

        # Left vs right
        left = rotated[:, :mw]
        right = np.fliplr(rotated[:, mw:])
        vert = iou(left, right)

        # Top vs bottom
        top = rotated[:mh, :]
        bottom = np.flipud(rotated[mh:, :])
        horz = iou(top, bottom)

        sym = (vert + horz) / 2.0
        if sym > best_sym:
            best_sym = sym
            if best_sym == 1.0:
                break
    return best_sym

# =====================================================================

