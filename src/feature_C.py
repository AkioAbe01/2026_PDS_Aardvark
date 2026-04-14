import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv
from skimage.segmentation import slic
from scipy.stats import circmean, circvar


def prepare_image_and_mask(image, mask):
    """
    The aim of this function is to make sure that the mask is boolean and RGB format is consistent for all images.

    ----
    First case: if the image is 2-dimensional, then convert to a 3D RGB by duplicating values. Also make sure that the 3rd dimension goest at the end.
                if there are 4 channels (RGBA) then drop the last channel
    Second case: if a mask has 3 dimensions we leave only the 1st channel

    Important check:
    Check whether every pixel in the mask has same spatial dimensions as a pixel in the image (and raise an error if not)

    ----
    Returns
        a cleaned RGB image
        a cleaned bool mask
    """
    #Image case
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)


    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    #Mask case
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = mask > 0

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image/mask mismatch: image={image.shape[:2]} mask={mask.shape}"
        )

    return image, mask


def slic_segmentation(image, mask, n_segments=30, compactness=10):
    """
    SLIC superpixel segmentation (only inside the lesion area)
    ---
    Converts image to a float and then applies the slic segmentation.
    parameters:
        n_segments- number of superpixels
        compactness- controls shape of segments
        start_label- labels start from 1 and not 0
        mask- cuts the image to take only the lesion
        channel_axis- shows that channels are in the last dimension
    Returns:
        a label map with same shape as the image and each pixel is a segment's ID
    """

    image_float = image.astype(np.float32) / 255.0

    segments = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=mask,
        channel_axis=-1
    )

    return segments


def compute_global_color_features(image, mask):
    """
    Extracts the overall color statistic of the whole lesion
    ----
    Computes global HSV statistics over the whole lesion.

    Cases:
        1. Handles an empty-mask case- set these to Nan
        2.Convert pixels from RGB to HSV
        3. Extract the HSV features: mean hue, mean saturation, mean value, hue variance, sautration and value standard dev
        (For mean hue: hue uses circular statistics since it wraps around)

    Returns:
        features: a dict, that contains 6 HSV statistics
        lesion_ppixels: RGB pixels inside the mask
    """
    lesion_pixels = image[mask]

    if len(lesion_pixels) == 0:
        features = {
            "hsv_mean_h": np.nan,
            "hsv_mean_s": np.nan,
            "hsv_mean_v": np.nan,
            "hsv_var_h":  np.nan,
            "hsv_std_s":  np.nan,
            "hsv_std_v":  np.nan,
        }
        return features, lesion_pixels

    hsv_pixels = rgb2hsv(lesion_pixels[np.newaxis, :, :] / 255.0)[0]

    features = {
        "hsv_mean_h": float(circmean(hsv_pixels[:, 0], high=1, low=0)),
        "hsv_mean_s": float(np.mean(hsv_pixels[:, 1])),
        "hsv_mean_v": float(np.mean(hsv_pixels[:, 2])),

        "hsv_var_h":  float(circvar(hsv_pixels[:, 0], high=1, low=0)),
        "hsv_std_s":  float(np.std(hsv_pixels[:, 1])),
        "hsv_std_v":  float(np.std(hsv_pixels[:, 2])),
    }

    return features, lesion_pixels


def compute_slic_color_features(image, mask, segments):
    """
    The main goal is to compute how much the lesion's color changes from one local region to the other
    in other words: it measures not just what color is the lesion but how UNEVENLY color is distributed inside the lesion
    ---
        convert an image to HSV
        take the superpixels that belong to the lesion only
        compute the mean HSV of each superpixel and measure how each of them vary accross the lesion
    ---
    Returns a dictionary consisting 3 features: hue variance, saturation std and value std across segments
    """
    image_float = image.astype(np.float32) / 255.0
    hsv_image = rgb2hsv(image_float)

    segment_ids = np.unique(segments[mask])

    hsv_means = []

    #loops through each lession superpixel and get mean HSV
    for seg_id in segment_ids:
        sp_mask = (segments == seg_id) & mask
        sp_pixels_hsv = hsv_image[sp_mask]

        if len(sp_pixels_hsv) == 0:
            continue

        seg_h_mean = circmean(sp_pixels_hsv[:, 0], high=1, low=0)
        seg_s_mean = np.mean(sp_pixels_hsv[:, 1])
        seg_v_mean = np.mean(sp_pixels_hsv[:, 2])

        hsv_means.append([seg_h_mean, seg_s_mean, seg_v_mean])

    # Case:there're no valid segments -> return Nan
    if len(hsv_means) == 0:
        return {
            "sp_hsv_var_h": np.nan,
            "sp_hsv_std_s": np.nan,
            "sp_hsv_std_v": np.nan,
        }

    hsv_means = np.array(hsv_means)

    # variability across segment-level HSV means
    #high variance means lesion has regions of very different colors which could be malignancy signal
    features = {
        "sp_hsv_var_h": float(circvar(hsv_means[:, 0], high=1, low=0)),
        "sp_hsv_std_s": float(np.std(hsv_means[:, 1])),
        "sp_hsv_std_v": float(np.std(hsv_means[:, 2])),
    }

    return features


def compute_relative_color_features(image, mask):
    """
    Measures how different the lesion color is from the skin around it
    (Since some lesions were almost the same color as the surrounding skin)
    ---
    Main task: compare lesion HSV color with the surrounding skin color

        convert the image to a float
        create two variables where: lesion_pixels-has the lesion pixels only, skin_pixels- has the skin pixels onl;y
        compute mean HSVs for both variables
        compute the hue differencr, satration and value differences
    ---
    Returns a dict thet has 3 features:  hue, saturation and value difference between lesion and skin
    """

    image_float = image.astype(np.float32) / 255.0

    lesion_pixels = image_float[mask]
    skin_pixels = image_float[~mask]

    # Handle in case lesion or skin is empty
    if len(lesion_pixels) == 0 or len(skin_pixels) == 0:
        return {
            "rel_hsv_diff_h": np.nan,
            "rel_hsv_diff_s": np.nan,
            "rel_hsv_diff_v": np.nan,
        }

    lesion_hsv = rgb2hsv(lesion_pixels[np.newaxis, :, :])[0]
    skin_hsv   = rgb2hsv(skin_pixels[np.newaxis, :, :])[0]

    lesion_h_mean = circmean(lesion_hsv[:, 0], high=1, low=0)
    lesion_s_mean = np.mean(lesion_hsv[:, 1])
    lesion_v_mean = np.mean(lesion_hsv[:, 2])

    skin_h_mean = circmean(skin_hsv[:, 0], high=1, low=0)
    skin_s_mean = np.mean(skin_hsv[:, 1])
    skin_v_mean = np.mean(skin_hsv[:, 2])

    hue_diff = abs(lesion_h_mean - skin_h_mean)
    hue_diff = min(hue_diff, 1 - hue_diff)  # wrap around since hue is circular

    features = {
        "rel_hsv_diff_h": float(hue_diff),
        "rel_hsv_diff_s": float(lesion_s_mean - skin_s_mean),
        "rel_hsv_diff_v": float(lesion_v_mean - skin_v_mean),
    }

    return features


def get_color_feature(image, mask, n_segments=30, compactness=10):
    """
    Final function that creates the full set of color features using previous functions (a total of 12 features)
    ---
    Returns a dictionary that contains:
        6 global lesion HSV statistics (mean and spread of HSV)
        3 SLIC-based regional HSV variation features
        3 relative lesion vs skin HSV difference features
    """

    image, mask = prepare_image_and_mask(image, mask)

    # --- global features ---
    global_features, lesion_pixels = compute_global_color_features(image, mask)

    # --- slic ---
    segments = slic_segmentation(
        image,
        mask,
        n_segments=n_segments,
        compactness=compactness
    )
    slic_features = compute_slic_color_features(image, mask, segments)

    # --- relative features ---
    relative_features = compute_relative_color_features(image, mask)

    # merge them all
    all_features = {}
    all_features.update(global_features)
    all_features.update(slic_features)
    all_features.update(relative_features)

    return all_features