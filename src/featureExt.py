from feature_A import modified_mask, lesion_symmetry, crop_to_bbox
from feature_B import get_border_feature
from feature_D import get_diameter_feature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from skimage.transform import resize
import time

# ======================================================================

def load_and_preprocess(img_id, max_dim=256):
    mask_path = f'../data/masks/{img_id.replace(".png", "_mask.png")}'
    raw = plt.imread(mask_path)

    if raw.ndim == 3:
        raw = raw[:,:,0]

    mask = modified_mask(raw).astype(bool)

    # Downsample if larger than max_dim
    h, w = mask.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        mask = resize(mask, (new_h, new_w), order=0, preserve_range=True).astype(bool)

    return crop_to_bbox(mask)   # final tight bounding box

# ---------- Main execution (must be guarded) ----------
def main(save = False):
    # Load your DataFrame – adjust as needed
    df = pd.read_csv('../data/new_metadata.csv', index_col = 0)
    df = df[df['Valid_mask'] == True]
    feature_df = df.loc[:,['patient_id', 'img_id', 'diagnostic']]
    features = ['feature_A', 'feature_B', 'feature_C', 'feature_D']

    for f in features:
        feature_df[f] = 0

    # Preprocess all masks once 
    mask_cache = {}
    for img_id in feature_df['img_id']:
        mask_cache[img_id] = load_and_preprocess(img_id, max_dim=256)

    # Create list of masks in DataFrame order
    mask_list = [mask_cache[id] for id in feature_df['img_id']]

    # Parallel computation
    with Pool(processes=8) as pool:
        symmetry_scores = pool.map(lesion_symmetry, mask_list)
        border_scores = pool.map(get_border_feature, mask_list)
        diameter_scores = pool.map(get_diameter_feature, mask_list)

    feature_df['feature_A'] = symmetry_scores
    feature_df['feature_B'] = border_scores
    feature_df['feature_D'] = diameter_scores

    if save:
        feature_df.to_csv('featureDf.csv')

    print(feature_df.head())
    print()
    print(feature_df.tail())

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Running time (min): {round(((time.time() - start) / 60), 2)}')
# ======================================================================