"""
models/xgb_model.py
==============================================================================
XGBoost baseline model for fraud detection.

Used as a strong gradient-boosted-trees comparison point against the DNN,
and also re-used at serving time in backend/predictor.py (its SHAP
TreeExplainer powers the live /api/predict explanations).
==============================================================================
"""

import xgboost as xgb


def build_xgb_model():
    """Construct the XGBoost classifier with the project's tuned hyperparameters."""
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # SMOTE already balanced the data
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )


def train_xgb(xgb_model, X_train_smote, y_train_smote):
    """Fit the XGBoost model on the SMOTE-balanced training data."""
    print("\nTraining XGBoost (Gradient Boosted Trees)...")
    xgb_model.fit(X_train_smote, y_train_smote)
    return xgb_model


def predict_xgb(xgb_model, X):
    """Fraud-class probability predictions for a batch of scaled features."""
    return xgb_model.predict_proba(X)[:, 1]
