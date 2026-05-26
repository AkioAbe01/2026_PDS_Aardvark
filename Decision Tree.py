import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
import joblib
import os

# ── 1. Load features ──────────────────────────────────────────────────
df = pd.read_csv('./featureDf_last.csv')

# ── 2. Define feature columns ─────────────────────────────────────────
color_cols = ['hsv_mean_h', 'hsv_mean_s', 'hsv_mean_v',
              'hsv_std_h', 'hsv_std_s', 'hsv_std_v',
              'sp_hsv_std_h', 'sp_hsv_std_s', 'sp_hsv_std_v',
              'rel_hsv_diff_h', 'rel_hsv_diff_s', 'rel_hsv_diff_v']

# ── 3. Binary cancer label ────────────────────────────────────────────
cancerous = ['BCC', 'MEL', 'SCC']
df['label'] = df['diagnostic'].isin(cancerous).astype(int)

# ── 4. Define X, y, groups ────────────────────────────────────────────
feature_cols = ['symmetry', 'border', 'diameter'] + color_cols
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
}).to_csv('./results/predictions/cv_folds_DT.csv', index=False)
print(f"CV fold scores saved to results/predictions/cv_folds_DT.csv")

# ── 7. Train final model ──────────────────────────────────────────────
final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)

# ── 8. Save model ─────────────────────────────────────────────────────
os.makedirs('./results/models', exist_ok=True)
joblib.dump(final_model, './results/models/decision_tree.pkl')

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
results_df.to_csv('./results/predictions/predictions_DT.csv', index=False)
print("\nPredictions saved to results/predictions/predictions_DT.csv")