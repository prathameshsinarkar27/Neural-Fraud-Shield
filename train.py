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

