import numpy as np
import pandas as pd
import cv2
# ======================================================================
# hair mask detect function 

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

# ======================================================================

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
def calculate_hair_coverage(img_id):
    """
    Calculate probability of lesion covered by hair.
    
    Args:
        image: RGB image
    Returns:
        coverage_probability: float (0-1)

    """
    img_path = '../data/imgs/' + img_id
    mask_path = '../data/masks/' + img_id.replace('.png', '_mask.png')

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0.5  # bool values

    # Get hair mask within lesion
    hair_mask = detect_hair_mask(image).astype(bool)
    
    # Count pixels (area)
    total_area = np.sum(mask) # lesion region
    hair_area = np.sum(hair_mask & mask) # hair in the lesion
    coverage = (hair_area) / (total_area)
    
    return round(coverage, 4)

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