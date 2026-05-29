"""
Feature-extraction entry point.

Runs the final feature-extraction pipeline (baseline: no hair removal, no
downscaling) over every image in DATA_DIR and writes the resulting feature
table to OUTPUT_CSV.

This is the canonical "features.csv" consumed by `main.py`.

TA workflow (no additional code needed):
  1. Point DATA_DIR below to the new dataset (it must contain `imgs/`, `masks/`,
     and `new_metadata.csv` in the same layout as the original `data/` folder).
  2. Run: `python src/extract_features.py`
     This regenerates OUTPUT_CSV based on the new images.
  3. Continue with `main.py` (set LOAD_MODEL = True there).

The feature definitions themselves live in:
  - src/feature_A.py  (asymmetry)
  - src/feature_B.py  (border compactness)
  - src/feature_C.py  (colour features)
  - src/feature_D.py  (diameter)
  - src/extract_features_baseline.py  (orchestrator: load image+mask, call A/B/C/D)
"""
import argparse
from extract_features_baseline import Config, run_feature_extraction, logger


# ---------------------------------------------------------------------------
# Paths -- edit these to point to a different dataset
# ---------------------------------------------------------------------------
DATA_DIR   = 'data'                  # must contain imgs/, masks/, new_metadata.csv
OUTPUT_CSV = 'data/features.csv'     # destination feature table


def main():
    # Allow command-line overrides for convenience, but the defaults above
    # are what a TA would normally edit.
    parser = argparse.ArgumentParser(
        description='Extract handcrafted ABCD features from dermoscopic images.',
    )
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help=f'Path to data directory (default: {DATA_DIR})')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                        help=f'Output CSV filename (default: {OUTPUT_CSV})')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes '
                             '(default: 75%% of CPU cores)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    args = parser.parse_args()

    config = Config(
        data_dir=args.data_dir,
        n_processes=args.processes,
    )

    try:
        run_feature_extraction(
            config=config,
            save_path=args.output,
            use_parallel=not args.no_parallel,
        )
        logger.info('Feature extraction completed successfully!')
    except Exception as e:
        logger.error(f'Feature extraction failed: {e}')
        raise


if __name__ == '__main__':
    main()
