import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from skimage.transform import resize
import time

# Import your custom modules – adjust paths as needed
from feature_A import modified_mask, lesion_symmetry, crop_to_bbox
from feature_B import get_border_feature
from feature_D import get_diameter_feature
from hair import remove_hair, remove_pen_mark
from colour import extract_all_colour_features       # 12‑colour‑feature function

# ----------------------------------------------------------------------
# 1. Load & preprocess (with optional background zeroing)
# ----------------------------------------------------------------------
def load_and_preprocess(img_id, max_dim=256, pen_mark=False, hair_rating=0):
    mask_path = f'../data/masks/{img_id.replace(".png", "_mask.png")}'
    img_path = f'../data/imgs/{img_id}'

    raw = plt.imread(mask_path)
    img = plt.imread(img_path)                     # uint8, RGB

    if raw.ndim == 3:
        raw = raw[:,:,0]

    mask = modified_mask(raw).astype(bool)

    h, w = mask.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        mask = resize(mask, (new_h, new_w), order=0, preserve_range=True).astype(bool)
        img = resize(img, (new_h, new_w), order=1, preserve_range=True)
        img = img.astype(np.uint8)                 # ← convert back to uint8

    mask, img = crop_to_bbox(mask, img)  # ensure mask, img are same shape

    # Ensure image is uint8 (OpenCV requirement)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
        
    # Hair removal (OpenCV expects uint8)
    if hair_rating == 1:
        img = remove_hair(img, radius=2, ksize=4)
    elif hair_rating == 2:
        img = remove_hair(img, radius=3, ksize=7)
    elif hair_rating == 3:
        img = remove_hair(img, radius=5, ksize=9)

    if pen_mark:
        img = remove_pen_mark(img)

    return mask, img


# ----------------------------------------------------------------------
# 2. Process a single image – all features in one pass
# ----------------------------------------------------------------------
def process_one_image(args):
    """
    args = (img_id, pen_mark, hair_rating)
    Returns a dictionary with all 12+3 features.
    """
    img_id, pen_mark, hair_rating = args
    
    # Load with background intact (for colour features)
    mask, img = load_and_preprocess(img_id, max_dim=256, pen_mark=pen_mark,
                                    hair_rating=hair_rating)
    
    # Colour features (uses RGB image & bool mask)
    colour_feat = extract_all_colour_features(img, mask)
    
    # Morphological features (mask only)
    symmetry = lesion_symmetry(mask)
    border   = get_border_feature(mask)
    diameter = get_diameter_feature(mask)
    
    # Combine everything
    result = {
        'img_id': img_id,
        **colour_feat,
        'symmetry': symmetry,
        'border': border,
        'diameter': diameter
    }
    return result


# ----------------------------------------------------------------------
# 3. Main: read data, prepare arguments, run parallel, save results
# ----------------------------------------------------------------------
def main(save_path='featureDf_last.csv'):
    print("Loading DataFrame...")
    df = pd.read_csv('../data/new_metadata.csv', index_col=0)
    df = df[df['Valid_mask'] == True]
    img_ids = df['img_id'].tolist()
    print(f"Total images to process: {len(img_ids)}")
    
    # Load annotation file (contains pen_rating, hair_rating per img_id)
    df_ann = pd.read_csv('../data/annotations.csv')
    ann_dict = {}
    for _, row in df_ann.iterrows():
        ann_dict[row['img_id']] = (row['pen_rating'] > 0, int(row['hair_rating']))
    
    # Build argument list for each image
    args_list = []
    missing_ann = []
    for img_id in img_ids:
        if img_id in ann_dict:
            pen, hair = ann_dict[img_id]
        else:
            pen, hair = False, 0
            missing_ann.append(img_id)
        args_list.append((img_id, pen, hair))
    
    if missing_ann:
        print(f"Warning: {len(missing_ann)} images missing annotations → using defaults (pen=False, hair=0)")
    
    # Parallel processing
    print("Starting parallel feature extraction (using all available cores)...")
    start_time = time.time()
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_one_image, args_list), total=len(args_list)))
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(results)
    
    # Save
    feature_df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")
    print(f"Time taken: {(time.time() - start_time)/60:.2f} minutes")
    
    # Show first few rows
    print("\nPreview:")
    print(feature_df.head())
    print(f"\nFeature columns: {list(feature_df.columns)}")


# ----------------------------------------------------------------------
# 4. Entry point
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()