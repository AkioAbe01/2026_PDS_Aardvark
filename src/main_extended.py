import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from skimage.transform import resize
import time
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings

# Add project root to sys.path so `from hair.hair import ...` works
# regardless of where this script is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import custom modules
from feature_A import modified_mask, lesion_symmetry, crop_to_bbox
from feature_B import get_border_feature
from feature_D import get_diameter_feature
from feature_C import extract_all_colour_features
from hair.hair import remove_hair, remove_pen_mark

# Original version (without resizing)

# ----------------------------------------------------------------------
# Configuration and Logging Setup
# ----------------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress scikit-image warnings about low-contrast images
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


class Config:
    """Configuration class for paths and parameters"""
    def __init__(self, 
                 data_dir: str = '../data',
                 max_dim: int = 256,
                 n_processes: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'imgs'
        self.mask_dir = self.data_dir / 'masks'
        self.metadata_path = self.data_dir / 'new_metadata.csv'
        self.annotations_path = self.data_dir / 'annotations.csv'
        self.max_dim = max_dim
        
        # Use 75% of available cores by default (leave some for system)
        if n_processes is None:
            self.n_processes = max(1, int(cpu_count() * 0.75))
        else:
            self.n_processes = n_processes
    
    def validate(self) -> bool:
        """Validate that all required paths exist"""
        missing = []
        if not self.img_dir.exists():
            missing.append(str(self.img_dir))
        if not self.mask_dir.exists():
            missing.append(str(self.mask_dir))
        if not self.metadata_path.exists():
            missing.append(str(self.metadata_path))
        
        if missing:
            logger.error(f"Missing required paths: {missing}")
            return False
        
        logger.info(f"Configuration validated. Using {self.n_processes} processes.")
        return True


# ----------------------------------------------------------------------
# Image Loading and Preprocessing
# ----------------------------------------------------------------------

def load_and_preprocess(img_id: str, 
                       config: Config,
                       pen_mark: bool = False, 
                       hair_rating: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load and preprocess a single image.
    
    Args:
        img_id: Image filename
        config: Configuration object
        pen_mark: Whether to remove pen marks
        hair_rating: Hair removal intensity (0=none, 1-3=increasing strength)
    
    Returns:
        (mask, image) tuple, or (None, None) if loading fails
    """
    try:
        mask_path = config.mask_dir / img_id.replace(".png", "_mask.png")
        img_path = config.img_dir / img_id
        
        # Check if files exist
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            return None, None
        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            return None, None
        
        # Load images (rgb by default)
        raw_mask = plt.imread(str(mask_path))
        img = plt.imread(str(img_path))
        
        # Handle multi-channel masks
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]
        
        # Process mask
        mask = modified_mask(raw_mask).astype(bool)
        
        # Validate mask
        if mask.sum() == 0:
            logger.warning(f"Empty mask for {img_id}")
            return None, None
        
        # Crop to bounding box
        mask, img = crop_to_bbox(mask, img)
        
        # Ensure uint8 for OpenCV operations
        if img.dtype != np.uint8:
            # Handle both normalized [0,1] and [0,255] ranges
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Drop alpha channel if present, and make the array C-contiguous —
        # cv2.inpaint rejects non-contiguous slices (the crop above is a view).
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.ascontiguousarray(img)

        # Hair removal based on rating
        if hair_rating == 1:
            img = remove_hair(img, radius=2, ksize=4)
        elif hair_rating == 2:
            img = remove_hair(img, radius=3, ksize=7)
        elif hair_rating == 3:
            img = remove_hair(img, radius=5, ksize=9)
        elif hair_rating > 3:
            logger.warning(f"Invalid hair_rating {hair_rating} for {img_id}, skipping hair removal")
        
        # Pen mark removal
        if pen_mark:
            img = remove_pen_mark(img)
        
        return mask, img
    
    except Exception as e:
        logger.error(f"Error loading/preprocessing {img_id}: {str(e)}")
        return None, None


# ----------------------------------------------------------------------
# Feature Extraction
# ----------------------------------------------------------------------

def process_one_image(args: Tuple[str, Config, bool, int]) -> Dict:
    """
    Extract all features from a single image.
    
    Args:
        args: Tuple of (img_id, config, pen_mark, hair_rating)
    
    Returns:
        Dictionary with img_id and all extracted features
    """
    img_id, config, pen_mark, hair_rating = args
    
    # Initialize result with img_id and NaN values
    result = {
        'img_id': img_id,
        'processing_status': 'failed',
        'error_message': None
    }

    try:
        # Load and preprocess
        mask, img = load_and_preprocess(img_id, config, pen_mark, hair_rating)
        
        if mask is None or img is None:
            result['error_message'] = 'Failed to load image or mask'
            return result
        
        # Extract color features (12 features)
        try:
            colour_feat = extract_all_colour_features(img, mask)
        except Exception as e:
            logger.error(f"Color feature extraction failed for {img_id}: {str(e)}")
            colour_feat = {f'color_feat_{i}': np.nan for i in range(12)}
        
        # Extract morphological features
        try:
            symmetry = lesion_symmetry(mask)
        except Exception as e:
            logger.error(f"Symmetry calculation failed for {img_id}: {str(e)}")
            symmetry = np.nan
        
        try:
            border = get_border_feature(mask)
        except Exception as e:
            logger.error(f"Border feature failed for {img_id}: {str(e)}")
            border = np.nan
        
        try:
            diameter = get_diameter_feature(mask)
        except Exception as e:
            logger.error(f"Diameter feature failed for {img_id}: {str(e)}")
            diameter = np.nan
        
        # Combine all features
        result = {
            'img_id': img_id,
            **colour_feat,
            'symmetry': symmetry,
            'border': border,
            'diameter': diameter,
            'processing_status': 'success',
            'error_message': None
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error processing {img_id}: {str(e)}")
        result['error_message'] = str(e)
        return result


# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------

def load_annotations(config: Config) -> Dict[str, Tuple[bool, int]]:
    """
    Load annotation data mapping img_id to (pen_mark, hair_rating).
    
    Returns:
        Dictionary mapping img_id to (pen_mark, hair_rating)
    """
    ann_dict = {}
    
    if not config.annotations_path.exists():
        logger.warning(f"Annotations file not found: {config.annotations_path}")
        logger.warning("All images will use default values (pen=False, hair=0)")
        return ann_dict
    
    try:
        df_ann = pd.read_csv(config.annotations_path)
        
        # Validate required columns
        required_cols = ['img_id', 'pen_rating', 'hair_rating']
        missing_cols = [col for col in required_cols if col not in df_ann.columns]
        if missing_cols:
            logger.error(f"Missing columns in annotations: {missing_cols}")
            return ann_dict
        
        for _, row in df_ann.iterrows():
            pen_mark = row['pen_rating'] > 0
            hair_rating = int(row['hair_rating'])
            ann_dict[row['img_id']] = (pen_mark, hair_rating)
        
        logger.info(f"Loaded annotations for {len(ann_dict)} images")
        return ann_dict
    
    except Exception as e:
        logger.error(f"Error loading annotations: {str(e)}")
        return ann_dict


def prepare_arguments(config: Config) -> Tuple[List, Dict]:
    """
    Prepare argument list for parallel processing.
    
    Returns:
        (args_list, stats) where args_list contains tuples for process_one_image
        and stats contains information about the preparation
    """
    # Load metadata
    logger.info("Loading metadata...")
    df = pd.read_csv(config.metadata_path, index_col=0)
    
    # Filter valid masks
    if 'Valid_mask' in df.columns:
        df = df[df['Valid_mask'] == True]
        logger.info(f"Filtered to {len(df)} images with valid masks")
    
    img_ids = df['img_id'].tolist()
    
    # Load annotations
    ann_dict = load_annotations(config)
    
    # Prepare arguments
    args_list = []
    missing_ann = []
    
    for img_id in img_ids:
        if img_id in ann_dict:
            pen, hair = ann_dict[img_id]
        else:
            pen, hair = False, 0
            missing_ann.append(img_id)
        
        args_list.append((img_id, config, pen, hair))
    
    stats = {
        'total_images': len(img_ids),
        'with_annotations': len(img_ids) - len(missing_ann),
        'missing_annotations': len(missing_ann)
    }
    
    if missing_ann:
        logger.warning(f"{len(missing_ann)} images missing annotations (using defaults)")
    
    return args_list, stats


def run_feature_extraction(config: Config, 
                          save_path: str = 'featureDf_extended.csv',
                          use_parallel: bool = True) -> pd.DataFrame:
    """
    Main feature extraction pipeline.
    
    Args:
        config: Configuration object
        save_path: Path to save the output CSV
        use_parallel: Whether to use parallel processing
    
    Returns:
        DataFrame with extracted features
    """
    # Validate configuration
    if not config.validate():
        raise ValueError("Configuration validation failed")
    
    # Prepare arguments
    args_list, stats = prepare_arguments(config)
    logger.info(f"Processing {stats['total_images']} images")
    
    # Process images
    start_time = time.time()
    
    if use_parallel and len(args_list) > 1:
        logger.info(f"Starting parallel processing with {config.n_processes} processes...")
        with Pool(processes=config.n_processes) as pool:
            # Use imap for memory efficiency (doesn't load all results at once)
            results = []
            for result in tqdm(pool.imap(process_one_image, args_list), 
                             total=len(args_list),
                             desc="Processing images"):
                results.append(result)
    else:
        logger.info("Processing images sequentially...")
        results = []
        for args in tqdm(args_list, desc="Processing images"):
            results.append(process_one_image(args))
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(results)
    
    # Report processing statistics
    success_count = (feature_df['processing_status'] == 'success').sum()
    failure_count = len(feature_df) - success_count
    
    logger.info(f"\nProcessing Summary:")
    logger.info(f"  Total images: {len(feature_df)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failure_count}")
    
    if failure_count > 0:
        logger.warning(f"Failed images:")
        failed = feature_df[feature_df['processing_status'] == 'failed']
        for _, row in failed.iterrows():
            logger.warning(f"  {row['img_id']}: {row['error_message']}")
    
    # Save results
    feature_df.to_csv(save_path, index=False)
    logger.info(f"\nResults saved to: {save_path}")
    
    # Timing
    elapsed = time.time() - start_time
    logger.info(f"Total processing time: {elapsed/60:.2f} minutes")
    logger.info(f"Average time per image: {elapsed/len(feature_df):.2f} seconds")
    
    # Show preview
    logger.info("\nFeature columns:")
    logger.info(f"{list(feature_df.columns)}")
    
    logger.info("\nFirst few rows:")
    print(feature_df.head())
    
    return feature_df


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

def main():
    """Entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from dermoscopy images')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Path to data directory (default: ../data)')
    parser.add_argument('--output', type=str, default='featureDf_extended.csv',
                       help='Output CSV filename (default: featureDf_extended.csv)')
    parser.add_argument('--max-dim', type=int, default=256,
                       help='Maximum image dimension (default: 256)')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: 75%% of CPU cores)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        data_dir=args.data_dir,
        max_dim=args.max_dim,
        n_processes=args.processes
    )
    
    # Run extraction
    try:
        feature_df = run_feature_extraction(
            config=config,
            save_path=args.output,
            use_parallel=not args.no_parallel
        )
        logger.info("Feature extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()