import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# ======================================================================
# READ IMAGE AND MASK
# ======================================================================
def read_img_mask(img_id, img_dir = '../data/imgs/', mask_dir = '../data/masks/'):
    '''
    Read the image given its img_id, img_dir and mask_dir. Ensures img and mask are correct.
    '''
    img_path = img_dir + img_id
    mask_path = mask_dir + img_id.replace('.png', '_mask.png')

    # Verify if path exists
    if (not os.path.exists(img_path)) or (not os.path.exists(mask_path)):
        print('Path Error')
        return np.nan, np.nan
    
    # Read Image and Mask
    img_bgr = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Check the correctness of img and mask
    if img_bgr.shape[:2] != mask.shape:
        print('Img-Mask mismatch Error')
        return np.nan, np.nan
    if img_bgr.shape[2] > 3: 
        # only keep the first 3 channel for the image
        img_bgr = img_bgr[:,:,:3]
    if mask.ndim > 2:
        # remove 3 dim is there is any
        mask = mask[:,:,0]
    if np.max(mask) > 1:
        # normalise its value [0,1]
        mask = (mask) / 255
    # binarise the mask
    binary_mask = np.where(mask > 0.5, 1, 0)
    return img_bgr, binary_mask

# ======================================================================
# HAIR FUNCTIONS
# ======================================================================

def detect_hair_mask(image, ksize = 3):
    """
    Detect hair (black, white, or mixed) in dermoscopy image.
    
    Args:
        image: RGB image (numpy array)
        ksize = Kernel size, default is 3
    Returns:
        hair_mask: Binary mask where hair pixels = 255
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Define kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
    
    # Detect dark (black) hair
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, black_hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Detect light (white) hair
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, white_hair_mask = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
    
    # Combine both masks (handles mixed hair automatically)
    combined_hair_mask = cv2.bitwise_or(black_hair_mask, white_hair_mask)
    
    return combined_hair_mask

# hair removal function
def remove_hair(img, radius = 3, ksize = 3):
    """
    Remove hair using inpainting.
    
    Args:
        image: RGB image
        radius: Default is 3. 
    Returns:
        inpainted_image: RGB image with hair removed
    """

    # ----- Force correct data type and channels -----
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # Handle number of channels
    if img.ndim == 2:
        # Grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        # RGBA -> BGR (drop alpha)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        # Already 3-channels – keep as is (OpenCV expects BGR, but inpainting works on RGB too)
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    hair_mask = detect_hair_mask(img, ksize = ksize)
    
    # Inpaint using the combined hair mask
    inpainted = cv2.inpaint(img, hair_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return inpainted

# hair coverage function

def calculate_hair_coverage(img_id: str):
    """
    Calculate probability of lesion covered by hair.
    
    Args:
        image_id: provide image id 
    Returns:
        coverage_probability: float (0-1)

    """
    if not isinstance(img_id, str): # expects string 
        return np.nan
    
    image_bgr, mask = read_img_mask(img_id) # img_bgr, binary_mask

    # Get hair mask within lesion
    hair_mask = detect_hair_mask(image_bgr)

    if hair_mask.shape != mask.shape:
        print('Mismatch Error in hair mask and lesion mask')
        return np.nan
    
    # Count pixels (area)

    lesion_area = np.sum(mask > 0) # lesion area
    hair_on_lesion = np.logical_and(hair_mask > 0, mask > 0)
    coverage = np.sum(hair_on_lesion) / lesion_area if lesion_area > 0 else 0 
    
    return round(coverage, 4)

# ======================================================================
# PEN MARK FUNCTIONS
# ======================================================================

# pen mask detect function

def create_blue_pen_mask(img):

    # Convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Typical wide blue range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create mask for blue range
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    
    return mask

# blue pen mark removal function
def remove_pen_mark(img, radius = 3):
    """
    Removes blue pen marks
    
    Args:
        image: RGB image
        radius: Default if 3.
    Returns:
        clean rgb image without pen_mark
    """
    pen_mask = create_blue_pen_mask(img)
    inpainted = cv2.inpaint(img, pen_mask, inpaintRadius = radius, flags = cv2.INPAINT_TELEA)
    return inpainted

# ======================================================================