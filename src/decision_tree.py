import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
import joblib
import os

# ── 0. Parse CLI arguments ────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train Decision Tree on extracted skin-lesion features')
parser.add_argument('--input', default='./featureDf_baseline.csv',
                    help='Path to feature CSV (default: ./featureDf_baseline.csv)')
parser.add_argument('--tag', default='baseline',
                    help='Tag appended to output filenames (baseline / extended / open_question)')
args = parser.parse_args()

# ── 1. Load features (all features already in this CSV) ───────────────
df = pd.read_csv(args.input)

# ── 2. Add diagnostic + patient_id from metadata ──────────────────────
meta = pd.read_csv('./data/new_metadata.csv')
df = df.merge(meta[['img_id', 'patient_id', 'diagnostic']], on='img_id', how='left')
df = df.dropna(subset=['patient_id', 'diagnostic'])  # need both for split + label

# ── 3. Binary cancer label ────────────────────────────────────────────
cancerous = ['BCC', 'MEL', 'SCC']
df['label'] = df['diagnostic'].isin(cancerous).astype(int)

# ── 4. Define X, y, groups (auto-detect feature columns) ──────────────
exclude = ['img_id', 'patient_id', 'diagnostic', 'label',
           'processing_status', 'error_message']
feature_cols = [c for c in df.columns if c not in exclude]
print(f"Input: {args.input} | tag: {args.tag} | {len(feature_cols)} features")

X = df[feature_cols].fillna(0)
y = df['label']
groups = df['patient_id']

# ── 5. Patient-level train/test split ─────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Features used: {feature_cols}")

# ── 6. Cross-validation to find best depth (on train only) ────────────
gkf = GroupKFold(n_splits=5)
best_depth, best_auc = None, 0
fold_scores_per_hp = {}

for depth in [2, 4, 6, 8, 10]:
    auc_scores = []
    for fold_train, fold_val in gkf.split(X_train, y_train, groups_train):
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
        probs = model.predict_proba(X_train.iloc[fold_val])[:, 1]
        auc_scores.append(roc_auc_score(y_train.iloc[fold_val], probs))
    fold_scores_per_hp[depth] = auc_scores
    mean_auc = np.mean(auc_scores)
    print(f"Depth {depth}: AUC = {mean_auc:.4f} ± {np.std(auc_scores):.4f}")
    if mean_auc > best_auc:
        best_auc, best_depth = mean_auc, depth

print(f"\nBest depth: {best_depth}")

# Save per-fold AUC scores for the best hyperparameter (used by evaluation.ipynb)
os.makedirs('./results/predictions', exist_ok=True)
best_fold_scores = fold_scores_per_hp[best_depth]
pd.DataFrame({
    'fold': range(1, len(best_fold_scores) + 1),
    'auc':  best_fold_scores,
    'hyperparam': best_depth,
}).to_csv(f'./results/predictions/cv_folds_DT_{args.tag}.csv', index=False)
print(f"CV fold scores saved to results/predictions/cv_folds_DT_{args.tag}.csv")

# ── 7. Train final model ──────────────────────────────────────────────
final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)

# ── 8. Save model ─────────────────────────────────────────────────────
os.makedirs('./results/models', exist_ok=True)
joblib.dump(final_model, f'./results/models/decision_tree_{args.tag}.pkl')

# ── 9. Evaluate on test set ───────────────────────────────────────────
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print(f"\n── Test Results ──────────────────")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")

# ── 10. Save predictions ──────────────────────────────────────────────
os.makedirs('./results/predictions', exist_ok=True)
results_df = df.iloc[test_idx][['patient_id', 'img_id', 'diagnostic', 'label']].copy()
results_df['predicted'] = y_pred
results_df['probability'] = y_prob
results_df.to_csv(f'./results/predictions/predictions_DT_{args.tag}.csv', index=False)
print(f"\nPredictions saved to results/predictions/predictions_DT_{args.tag}.csv")