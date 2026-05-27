import pandas as pd
import numpy as np
import os
import argparse
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

# ── 0. Parse CLI arguments ──────────────────────────────────
parser = argparse.ArgumentParser(description='Train kNN on extracted skin-lesion features')
parser.add_argument('--input', default='./featureDf_baseline.csv',
                    help='Path to feature CSV (default: ./featureDf_baseline.csv)')
parser.add_argument('--tag', default='baseline',
                    help='Tag appended to output filenames (baseline / extended / open_question)')
args = parser.parse_args()

# ── 1. Load features (all features already in this CSV) ─────
df = pd.read_csv(args.input)

# ── 2. Add diagnostic + patient_id from metadata ────────────
meta = pd.read_csv('./data/new_metadata.csv')
df = df.merge(meta[['img_id', 'patient_id', 'diagnostic']], on='img_id', how='left')
df = df.dropna(subset=['patient_id', 'diagnostic'])  # need both for split + label

# ── 3. Create label ─────────────────────────────────────────
cancerous = ['BCC', 'MEL', 'SCC']
df['label'] = df['diagnostic'].isin(cancerous).astype(int)

# ── 4. Define data (auto-detect feature columns) ────────────
exclude = ['img_id', 'patient_id', 'diagnostic', 'label',
           'processing_status', 'error_message']
feature_cols = [c for c in df.columns if c not in exclude]
print(f"Input: {args.input} | tag: {args.tag} | {len(feature_cols)} features")

X = df[feature_cols]
y = df['label']
groups = df['patient_id']

# ── 5. Train/test split (patient-level) ─────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# ── 6. Tune k ───────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)

k_values = [3, 5, 7, 9, 11]
best_k, best_auc = None, 0
fold_scores_per_hp = {}

for k in k_values:
    auc_scores = []

    for fold_train, fold_val in gkf.split(X_train, y_train, groups_train):
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])

        model.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
        probs = model.predict_proba(X_train.iloc[fold_val])[:, 1]

        auc_scores.append(roc_auc_score(y_train.iloc[fold_val], probs))

    fold_scores_per_hp[k] = auc_scores
    mean_auc = np.mean(auc_scores)
    print(f"k={k}: AUC = {mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc, best_k = mean_auc, k

print(f"Best k: {best_k}")

# Save per-fold AUC scores for the best hyperparameter (used by evaluation.ipynb)
os.makedirs('./results/predictions', exist_ok=True)
best_fold_scores = fold_scores_per_hp[best_k]
pd.DataFrame({
    'fold': range(1, len(best_fold_scores) + 1),
    'auc':  best_fold_scores,
    'hyperparam': best_k,
}).to_csv(f'./results/predictions/cv_folds_kNN_{args.tag}.csv', index=False)
print(f"CV fold scores saved to results/predictions/cv_folds_kNN_{args.tag}.csv")

# ── 7. Train final model ────────────────────────────────────
final_knn = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])

final_knn.fit(X_train, y_train)

# ── 8. Evaluate ─────────────────────────────────────────────
y_pred = final_knn.predict(X_test)
y_prob = final_knn.predict_proba(X_test)[:, 1]

print("\n── kNN Test Results ───────────────")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")

# ── 9. Save model ───────────────────────────────────────────
os.makedirs('./results/models', exist_ok=True)
joblib.dump(final_knn, f'./results/models/knn_model_{args.tag}.pkl')

#save predicitons
os.makedirs('./results/predictions', exist_ok=True)
results_df = df.iloc[test_idx][['patient_id', 'img_id', 'diagnostic', 'label']].copy()
results_df['predicted'] = y_pred
results_df['probability'] = y_prob
results_df.to_csv(f'./results/predictions/predictions_kNN_{args.tag}.csv', index=False)
print(f"\nPredictions saved to results/predictions/predictions_kNN_{args.tag}.csv")