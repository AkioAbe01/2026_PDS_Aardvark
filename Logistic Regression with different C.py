import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ── 1. Load features ─────────────────────────────────────────
df = pd.read_csv('./src/featureDf.csv', index_col=0)

# ── 2. Merge color features ─────────────────────────────────
color_df = pd.read_csv('./data/features_data.csv')
color_df['img_id'] = color_df['image'].str.replace('data/imgs/', '', regex=False)

color_cols = ['hsv_mean_h', 'hsv_mean_s', 'hsv_mean_v',
              'hsv_var_h', 'hsv_std_s', 'hsv_std_v',
              'sp_hsv_var_h', 'sp_hsv_std_s', 'sp_hsv_std_v',
              'rel_hsv_diff_h', 'rel_hsv_diff_s', 'rel_hsv_diff_v']

df = df.merge(color_df[['img_id'] + color_cols], on='img_id', how='left')

# ── 3. Create label ─────────────────────────────────────────
cancerous = ['BCC', 'MEL', 'SCC']
df['label'] = df['diagnostic'].isin(cancerous).astype(int)

# ── 4. Define data ──────────────────────────────────────────
feature_cols = ['feature_A', 'feature_B', 'feature_D'] + color_cols

X = df[feature_cols]
y = df['label']
groups = df['patient_id']

# ── 5. Train/test split ─────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# ── 6. Tune C ───────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)

C_values = [0.01, 0.1, 1, 10, 100]
best_C, best_auc = None, 0

for C in C_values:
    auc_scores = []

    for fold_train, fold_val in gkf.split(X_train, y_train, groups_train):
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            ))
        ])

        model.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
        probs = model.predict_proba(X_train.iloc[fold_val])[:, 1]

        auc_scores.append(roc_auc_score(y_train.iloc[fold_val], probs))

    mean_auc = np.mean(auc_scores)
    print(f"C={C}: AUC = {mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc, best_C = mean_auc, C

print(f"Best C: {best_C}")

# ── 7. Train final model ────────────────────────────────────
final_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        C=best_C,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    ))
])

final_lr.fit(X_train, y_train)

# ── 8. Evaluate ─────────────────────────────────────────────
y_pred = final_lr.predict(X_test)
y_prob = final_lr.predict_proba(X_test)[:, 1]

print("\n── Logistic Regression Test Results ───")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")

# ── 9. Save model ───────────────────────────────────────────
os.makedirs('./results/models', exist_ok=True)
joblib.dump(final_lr, './results/models/logreg_model.pkl')
