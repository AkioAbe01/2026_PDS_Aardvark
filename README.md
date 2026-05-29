# Projects in Data Science 2026 — Group Aardvark

Skin-lesion (cancer vs. non-cancer) classification on PAD-UFES-20 dermoscopic
images using hand-crafted ABCD features.

## How to reproduce (TA workflow)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract features from images

```bash
python src/extract_features.py
```

This reads every image/mask from `data/imgs/` and `data/masks/` and writes the
feature table to `data/features.csv`. To point the script at a different
dataset, edit `DATA_DIR` / `OUTPUT_CSV` at the top of
`src/extract_features.py`.

### 3. Run the model

Open `main.py` and choose the mode at the top of the file:

- `LOAD_MODEL = True` (default): load the trained model from
  `results/models/final_model.pkl` and write predictions to
  `results/predictions/final_predictions.csv`. No retraining.
- `LOAD_MODEL = False`: perform 5-fold patient-level cross-validation, train
  Logistic Regression / kNN / Decision Tree, save each model under
  `results/models/`, save each model's predictions under
  `results/predictions/`, and save the best model as `final_model.pkl`.

Then:

```bash
python main.py
```

## Outputs

- `data/features.csv` - extracted features (produced by step 2)
- `results/models/final_model.pkl` - trained final model
- `results/predictions/final_predictions.csv` - predictions on the held-out
  test split (or full dataset if `LOAD_MODEL = True`)
- `results/figures/` - figures used in the report
- `results/reports/report_Aardvark.pdf` - written report
- `results/reports/features_Aardvark.csv` - copy of features for the report

## Source layout

- `src/extract_features.py` - TA entry point for feature extraction
- `src/extract_features_baseline.py` - orchestrator (no hair removal,
  no downscaling) used by `extract_features.py`
- `src/extract_features_extended.py` - pipeline with hair removal
- `src/extract_features_open_question.py` - pipeline with image downscaling
- `src/feature_A.py / feature_B.py / feature_C.py / feature_D.py` - per-feature
  implementations (asymmetry, border, colour, diameter)
- `src/hair.py` - hair-removal helpers
- `src/*.ipynb` - exploratory notebooks (EDA, feature comparison, model
  evaluation); not required to reproduce the deliverables

## Notes

This repo is the hand-in for the ITU course "Projects in Data Science" 2026,
group Aardvark.

#### File Hierarchy

The file hierarchy of your hand-in repo should be as follows:

```
ProjectInDataScience2026_ExamTemplate/
├── data/
│   ├─ features.csv                     # all image file names, ground-truth labels, and chosen features
│   ├─ annotations_combined.csv         # annotations of hair and penmarks
│   │
│   ├── imgs/                           # skin images (to not add on GitHub)
│   │    ├── img_XX1.png
│   │    ├── img_XX2.png
│   │     ......
│   │    └── img_XXX.png
│   │
│   └── masks/                          # masks images (to not add on GitHub)
│        ├── mask_XX1.png
│        ├── mask_XX2.png
│         ......
│        └── mask_XXX.png
│
├── src/
│   ├── __init__.py
│   ├── feature_A.py                    # code for feature A extraction
│   ├── feature_B.py                    # code for feature B extraction
│   ......
│   └── feature_X.py                    # code for feature X extraction
│ 
├── result/
│   ├── figures/                        # Figures used in your report
│   ├── models/                         # Trained models
│   ├── predictions/                    # Probabilities outputed by the models
│   └── reports                         # Files related to the Mandatory assignment
│        ├── report_GROUPEID.pdf
│        └── features_GROUPEID.csv
│ 
├── main.py                             # script to train or evaluate models
└── README.md
```

**Notes:**

1. DO NOT upload your data (images) to Github.
2. When the same code block needs to be executed multiple times in the script, make it a custom function instead. All the custom functions and modules should be grouped into different files under the *"src"* subfolder, based on the task they are designed for. Do not put everything in a single Python file or copy-paste the same code block across the script.
