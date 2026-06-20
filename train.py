"""
================================================================================
NNARL Project 1: Financial Fraud Detection using Deep Neural Networks - train.py
================================================================================
  STEP 1  Load Dataset                  -> data/preprocessing.py
  STEP 2  Feature Engineering           -> data/feature_engineering.py
  STEP 3  Train-Test Split & Scaling    -> data/preprocessing.py
  STEP 4  SMOTE Oversampling            -> data/preprocessing.py
  STEP 5  Build & Train the DNN         -> models/dnn_model.py
  STEP 6  Train XGBoost Baseline        -> models/xgb_model.py
  STEP 7  Compute Metrics               -> models/explainability.py
  STEP 8  t-SNE Visualization           -> (here; single use, kept inline)
  STEP 9  Feature Distribution          -> (here; single use, kept inline)
  STEP 10 Hourly / Amount Distribution  -> (here; single use, kept inline)
  STEP 11 SHAP Explainer                -> models/explainability.py
  STEP 12 LIME Explainer                -> (here; single use, kept inline)
==============================================================================
"""

import os
import json
import pickle
import warnings

import numpy as np

from data.preprocessing import load_dataset, split_and_scale, apply_smote
from data.feature_engineering import (
    FEATURE_COLS, engineer_features, compute_feature_stats, compute_feature_importance
)
from models.dnn_model import build_dnn_model, get_architecture_info, train_dnn, predict_dnn
from models.xgb_model import build_xgb_model, train_xgb, predict_xgb
from models.explainability import (
    compute_metrics, compute_curve, compute_pr_curve, compute_threshold_data,
    fit_shap_kernel_explainer, precompute_shap_samples
)

warnings.filterwarnings('ignore')

ARTIFACTS_DIR = "artifacts"


# ===================== STEP 0: Create output directory =====================
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ===================== STEP 1: Load Dataset ================================
print("=" * 70)
print("STEP 1: Loading the Credit Card Fraud Detection Dataset")
print("=" * 70)

df = load_dataset()

# Save class distribution for frontend
class_dist = df['Class'].value_counts().to_dict()
class_dist = {str(k): int(v) for k, v in class_dist.items()}

# ===================== STEP 2: Feature Engineering =========================
print("\n" + "=" * 70)
print("STEP 2: Feature Engineering")
print("=" * 70)

df = engineer_features(df)
feature_cols = FEATURE_COLS
X = df[feature_cols].values
y = df['Class'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Features used: {feature_cols}")

print("\nComputing feature statistics for translation layer...")
fraud_mask = y == 1
legit_mask = y == 0
feature_stats = compute_feature_stats(X, y, feature_cols)

with open(f"{ARTIFACTS_DIR}/feature_stats.json", "w") as f:
    json.dump(feature_stats, f, indent=2)
print(f"Feature statistics saved to {ARTIFACTS_DIR}/feature_stats.json")

feature_importance = compute_feature_importance(feature_stats, feature_cols)
with open(f"{ARTIFACTS_DIR}/feature_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=2)

# ===================== STEP 3: Train-Test Split & Scaling ==================
print("\n" + "=" * 70)
print("STEP 3: Train-Test Split & Standard Scaling")
print("=" * 70)

X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y, ARTIFACTS_DIR)

# ===================== STEP 4: SMOTE Oversampling ==========================
print("\n" + "=" * 70)
print("STEP 4: Applying SMOTE to Handle Class Imbalance")
print("=" * 70)

X_train_smote, y_train_smote, smote_data = apply_smote(X_train_scaled, y_train)

with open(f"{ARTIFACTS_DIR}/smote_data.json", "w") as f:
    json.dump(smote_data, f, indent=2)

# ===================== STEP 5: Build & Train the DNN =======================
print("\n" + "=" * 70)
print("STEP 5: Building the Deep Neural Network (DNN)")
print("=" * 70)

model = build_dnn_model()
architecture_info = get_architecture_info(model)
total_params = int(model.count_params())

history, best_epoch = train_dnn(model, X_train_smote, y_train_smote, X_test_scaled, y_test)

model.save(f"{ARTIFACTS_DIR}/dnn_model.h5")
print("DNN model saved.")

training_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
training_history['best_epoch'] = best_epoch
training_history['batch_size'] = 256
training_history['total_params'] = total_params
training_history['architecture'] = architecture_info
training_history['optimizer'] = {
    'name': 'Adam',
    'learning_rate': 0.001,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7
}
training_history['loss_function'] = 'Binary Cross-Entropy'
training_history['training_method'] = 'Backpropagation'

with open(f"{ARTIFACTS_DIR}/training_history.json", "w") as f:
    json.dump(training_history, f, indent=2)

# ===================== STEP 6: Baseline Model — XGBoost ===================
print("\n" + "=" * 70)
print("STEP 6: Training XGBoost Baseline Model")
print("=" * 70)

xgb_model = build_xgb_model()
xgb_model = train_xgb(xgb_model, X_train_smote, y_train_smote)
xgb_probs = predict_xgb(xgb_model, X_test_scaled)
xgb_preds = (xgb_probs >= 0.5).astype(int)

with open(f"{ARTIFACTS_DIR}/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print(f"XGBoost model saved to {ARTIFACTS_DIR}/xgb_model.pkl")

# DNN predictions
dnn_probs = predict_dnn(model, X_test_scaled)
dnn_preds = (dnn_probs >= 0.5).astype(int)

# ===================== STEP 7: Compute All Metrics =========================
print("\n" + "=" * 70)
print("STEP 7: Computing Metrics for All Models")
print("=" * 70)

dnn_metrics = compute_metrics(y_test, dnn_preds, dnn_probs, "DNN")
xgb_metrics = compute_metrics(y_test, xgb_preds, xgb_probs, "XGBoost")

all_metrics = {
    'dnn': dnn_metrics,
    'xgboost': xgb_metrics,
    'roc_curves': {
        'dnn': compute_curve(y_test, dnn_probs),
        'xgb': compute_curve(y_test, xgb_probs)
    },
    'pr_curves': {
        'dnn': compute_pr_curve(y_test, dnn_probs),
        'xgb': compute_pr_curve(y_test, xgb_probs)
    },
    'threshold_data': compute_threshold_data(y_test, dnn_probs),
    'class_distribution': class_dist,
    'smote': smote_data,
    'dataset_info': {
        'total_samples': int(len(df)),
        'total_fraud': int(class_dist.get('1', 0)),
        'total_legit': int(class_dist.get('0', 0)),
        'n_features': 30
    }
}

with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nAll metrics saved to {ARTIFACTS_DIR}/metrics.json")

# ===================== STEP 8: t-SNE for PCA Visualization =================
print("\n" + "=" * 70)
print("STEP 8: Computing t-SNE Embedding for PCA Space Visualization")
print("=" * 70)

from sklearn.manifold import TSNE

sample_size = 5000
np.random.seed(42)
idx = np.random.choice(len(X_test_scaled), min(sample_size, len(X_test_scaled)), replace=False)
X_sample = X_test_scaled[idx]
y_sample = y_test[idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(X_sample)

tsne_data = []
for i in range(len(tsne_result)):
    tsne_data.append({
        'x': round(float(tsne_result[i, 0]), 2),
        'y': round(float(tsne_result[i, 1]), 2),
        'label': int(y_sample[i])
    })

with open(f"{ARTIFACTS_DIR}/tsne_data.json", "w") as f:
    json.dump(tsne_data, f)
print(f"t-SNE data saved ({len(tsne_data)} points)")

# ===================== STEP 9: Feature Distribution Comparison =============
print("\n" + "=" * 70)
print("STEP 9: Feature Distribution Comparison")
print("=" * 70)

top_features = [f['feature'] for f in feature_importance[:10]]
distribution_data = {}
for feat in top_features:
    idx_f = feature_cols.index(feat)
    distribution_data[feat] = {
        'legit_mean': round(float(np.mean(X[legit_mask, idx_f])), 4),
        'fraud_mean': round(float(np.mean(X[fraud_mask, idx_f])), 4),
        'legit_std': round(float(np.std(X[legit_mask, idx_f])), 4),
        'fraud_std': round(float(np.std(X[fraud_mask, idx_f])), 4)
    }

with open(f"{ARTIFACTS_DIR}/distribution_data.json", "w") as f:
    json.dump(distribution_data, f, indent=2)

# ===================== STEP 10: Hourly Transaction Distribution ============
print("\n" + "=" * 70)
print("STEP 10: Hourly Transaction Volume Analysis")
print("=" * 70)

hourly_data = {}
hours = df['Hour'].astype(int)
for h in range(24):
    mask = hours == h
    hourly_data[str(h)] = {
        'total': int(mask.sum()),
        'fraud': int((df['Class'][mask] == 1).sum())
    }

with open(f"{ARTIFACTS_DIR}/hourly_data.json", "w") as f:
    json.dump(hourly_data, f, indent=2)

amount_buckets = [0, 1, 10, 50, 100, 500, 1000, 5000, 30000]
amount_dist = []
for i in range(len(amount_buckets) - 1):
    lo, hi = amount_buckets[i], amount_buckets[i + 1]
    mask = (df['Amount'] >= lo) & (df['Amount'] < hi)
    amount_dist.append({
        'bucket': f"${lo}-${hi}",
        'count': int(mask.sum()),
        'fraud_count': int((df['Class'][mask] == 1).sum())
    })

with open(f"{ARTIFACTS_DIR}/amount_distribution.json", "w") as f:
    json.dump(amount_dist, f, indent=2)

# ===================== STEP 11: SHAP Explainer =============================
print("\n" + "=" * 70)
print("STEP 11: Fitting SHAP Explainer")
print("=" * 70)

explainer, background = fit_shap_kernel_explainer(model, X_train_smote, background_size=100)

with open(f"{ARTIFACTS_DIR}/shap_background.pkl", "wb") as f:
    pickle.dump(background, f)

print("Precomputing SHAP values for sample test instances...")
shap_precomputed = precompute_shap_samples(explainer, model, X_test_scaled, feature_cols, n_samples=20)

with open(f"{ARTIFACTS_DIR}/shap_precomputed.json", "w") as f:
    json.dump(shap_precomputed, f, indent=2)
print("SHAP precomputed data saved.")

# ===================== STEP 12: LIME Explainer =============================
print("\n" + "=" * 70)
print("STEP 12: Setting up LIME Explainer")
print("=" * 70)

import lime
import lime.lime_tabular

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_smote,
    feature_names=feature_cols,
    class_names=['Legitimate', 'Fraud'],
    mode='classification'
)
print("LIME explainer created.")

np.save(f"{ARTIFACTS_DIR}/lime_training_data.npy", X_train_smote[:1000])
print("LIME training sample saved.")

with open(f"{ARTIFACTS_DIR}/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)


print("\n" + "=" * 70)
print("ALL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nArtifacts saved in ./{ARTIFACTS_DIR}/:")
for fname in sorted(os.listdir(ARTIFACTS_DIR)):
    fsize = os.path.getsize(f"{ARTIFACTS_DIR}/{fname}")
    print(f"  {fname} ({fsize:,} bytes)")
print("\nNext step: Run app.py to start the Flask API server.")
