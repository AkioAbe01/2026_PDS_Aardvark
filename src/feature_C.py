import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from scipy.stats import circmean

# ----------------------------------------------------------------------
# Helper: circular mean for hue (0-180 in OpenCV, but we'll work in [0,1])
# ----------------------------------------------------------------------
def circular_mean_hue(hue_vals_deg):
    """
    hue_vals_deg : array of hue values in degrees (0-180 for OpenCV)
    Returns circular mean in degrees (0-180)
    """
    # Convert to radians
    hue_rad = np.deg2rad(2 * hue_vals_deg)  # because hue is circular with period 180°
    mean_rad = np.arctan2(np.mean(np.sin(hue_rad)), np.mean(np.cos(hue_rad)))
    mean_deg = np.rad2deg(mean_rad) / 2.0
    if mean_deg < 0:
        mean_deg += 180
    return mean_deg

# ----------------------------------------------------------------------
# Group 1: Global HSV statistics of the lesion (6 features)
# ----------------------------------------------------------------------
def compute_global_color_features(image_bgr, mask):
    """
    image_bgr: BGR image (H,W,3) uint8
    mask: binary mask (H,W) where True = lesion
    returns: dict with global HSV features
    """
    # Convert to HSV (OpenCV: H 0-180, S 0-255, V 0-255)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_vals = hsv[mask, 0].astype(np.float64)
    s_vals = hsv[mask, 1].astype(np.float64)
    v_vals = hsv[mask, 2].astype(np.float64)

    if len(h_vals) == 0:
        return {k: np.nan for k in ['hsv_mean_h', 'hsv_mean_s', 'hsv_mean_v',
                                    'hsv_std_h', 'hsv_std_s', 'hsv_std_v']}

    # Circular mean for hue
    mean_h = circular_mean_hue(h_vals) / 180.0   # normalise to [0,1]
    mean_s = np.mean(s_vals) / 255.0
    mean_v = np.mean(v_vals) / 255.0

    # Standard deviations (for hue, use linear std as approximate – careful with wrap)
    std_h = np.std(h_vals) / 180.0
    std_s = np.std(s_vals) / 255.0
    std_v = np.std(v_vals) / 255.0

    return {
        'hsv_mean_h': mean_h,
        'hsv_mean_s': mean_s,
        'hsv_mean_v': mean_v,
        'hsv_std_h': std_h,
        'hsv_std_s': std_s,
        'hsv_std_v': std_v
    }

# ----------------------------------------------------------------------
# Group 2: Local colour non-uniformity via SLIC (3 features)
# ----------------------------------------------------------------------
def compute_slic_color_features(image_bgr, mask, n_segments=25, compactness=10):
    """
    Compute variance of mean HSV across superpixels inside the lesion.
    """
    # Convert BGR to RGB (SLIC expects RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_float = image_rgb.astype(np.float32) / 255.0

    # Apply SLIC
    superpixels = slic(img_float, n_segments=n_segments, compactness=compactness,
                       start_label=1, channel_axis=-1)

    # Restrict superpixels to lesion mask
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask
    superpixels[~mask_bool] = 0

    # Compute mean HSV for each superpixel (using HSV from BGR image)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    unique_labels = np.unique(superpixels)
    unique_labels = unique_labels[unique_labels > 0]

    means_h = []
    means_s = []
    means_v = []

    for label in unique_labels:
        sp_mask = (superpixels == label)
        h_vals = hsv[sp_mask, 0].astype(np.float64)
        s_vals = hsv[sp_mask, 1].astype(np.float64)
        v_vals = hsv[sp_mask, 2].astype(np.float64)

        if len(h_vals) < 10:
            continue
        # Circular mean for hue (per superpixel)
        mean_h = circular_mean_hue(h_vals) / 180.0
        means_h.append(mean_h)
        means_s.append(np.mean(s_vals) / 255.0)
        means_v.append(np.mean(v_vals) / 255.0)

    if len(means_h) == 0:
        return {'sp_hsv_std_h': np.nan, 'sp_hsv_std_s': np.nan, 'sp_hsv_std_v': np.nan}

    # Standard deviation of superpixel means = local non-uniformity
    return {
        'sp_hsv_std_h': np.std(means_h),
        'sp_hsv_std_s': np.std(means_s),
        'sp_hsv_std_v': np.std(means_v)
    }

# ----------------------------------------------------------------------
# Group 3: Contrast "lesion vs skin" (3 features)
# ----------------------------------------------------------------------
def compute_relative_color_features(image_bgr, mask, skin_margin=20):
    """
    Compare lesion colour to surrounding healthy skin.
    Skin region = dilated mask minus original mask.
    """
    # Ensure mask is uint8 (0/255) for morphological operations
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)
        if np.max(mask_uint8) == 1:
            mask_uint8 *= 255

    # Dilate mask to get a surrounding ring
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (skin_margin*2+1, skin_margin*2+1))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    skin_region = cv2.bitwise_and(dilated, cv2.bitwise_not(mask_uint8))

    if np.sum(skin_region) == 0:
        return {'rel_hsv_diff_h': np.nan, 'rel_hsv_diff_s': np.nan, 'rel_hsv_diff_v': np.nan}

    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Lesion statistics
    lesion_h = hsv[mask_uint8 > 0, 0].astype(np.float64)
    lesion_s = hsv[mask_uint8 > 0, 1].astype(np.float64)
    lesion_v = hsv[mask_uint8 > 0, 2].astype(np.float64)
    # Skin statistics
    skin_h = hsv[skin_region > 0, 0].astype(np.float64)
    skin_s = hsv[skin_region > 0, 1].astype(np.float64)
    skin_v = hsv[skin_region > 0, 2].astype(np.float64)

    if len(lesion_h) == 0 or len(skin_h) == 0:
        return {'rel_hsv_diff_h': np.nan, 'rel_hsv_diff_s': np.nan, 'rel_hsv_diff_v': np.nan}

    # Circular mean difference for hue (minimum angular distance)
    mean_h_lesion = circular_mean_hue(lesion_h)
    mean_h_skin = circular_mean_hue(skin_h)
    diff_h = min(abs(mean_h_lesion - mean_h_skin), 180 - abs(mean_h_lesion - mean_h_skin)) / 180.0

    # Simple difference for saturation and value
    mean_s_lesion = np.mean(lesion_s) / 255.0
    mean_s_skin = np.mean(skin_s) / 255.0
    diff_s = abs(mean_s_lesion - mean_s_skin)

    mean_v_lesion = np.mean(lesion_v) / 255.0
    mean_v_skin = np.mean(skin_v) / 255.0
    diff_v = abs(mean_v_lesion - mean_v_skin)

    return {
        'rel_hsv_diff_h': diff_h,
        'rel_hsv_diff_s': diff_s,
        'rel_hsv_diff_v': diff_v
    }

# ----------------------------------------------------------------------
# Master function: extract all colour features for one image
# ----------------------------------------------------------------------
def extract_all_colour_features(image_bgr, mask, slic_n_segments=100, slic_compactness=20, skin_margin=20):
    """
    Returns a dictionary with all 12 features (6+3+3).
    """
    features = {}
    features.update(compute_global_color_features(image_bgr, mask))
    features.update(compute_slic_color_features(image_bgr, mask, n_segments=slic_n_segments, compactness=slic_compactness))
    features.update(compute_relative_color_features(image_bgr, mask, skin_margin=skin_margin))
    return features

# ----------------------------------------------------------------------
# Example: apply to your DataFrame
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load a single image and mask
    img_id = 'PAT_597_1139_181.png'
    img_path = '../data/imgs/' + img_id
    mask_path = '../data/masks/' + img_id.replace('.png', '_mask.png')

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0

    feat = extract_all_colour_features(img, mask)
    print(pd.Series(feat))

    # To apply to all rows in df_ann:
    # for i, row in df_ann.iterrows():
    #     img = cv2.imread(f"{img_dir}/{row['img_id']}")
    #     mask = cv2.imread(f"{mask_dir}/{row['img_id']}", cv2.IMREAD_GRAYSCALE) > 0
    #     if img is not None and mask is not None:
    #         feat = extract_all_colour_features(img, mask)
    #         for k, v in feat.items():
    #             df_ann.at[i, k] = v