"""
data/feature_engineering.py
==============================================================================
Feature engineering for the Neural Fraud Shield.
  - Deriving Log_Amount and Hour from the raw Time/Amount columns
  - Defining the 30-feature column order used everywhere downstream
  - Computing per-feature statistics (mean/std for legit vs fraud) that
    power the translation layer in backend/translation_layer.py
  - Computing Pearson correlation-based feature importance for the frontend
==============================================================================
"""

import numpy as np

# Features for model: V1-V28 + Log_Amount + Hour = 30 features
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Log_Amount', 'Hour']


def engineer_features(df):
    """Add Log_Amount and Hour columns to the raw dataframe (in place + returned)."""
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] % 86400) / 3600  # Convert seconds to hour of day
    return df


def compute_feature_stats(X, y, feature_cols=FEATURE_COLS):
    """Per-feature legit/fraud mean, std, and correlation with the fraud label.

    Used by the translation layer to map human-readable UI inputs onto the
    30-dim PCA feature vector at inference time.
    """
    fraud_mask = y == 1
    legit_mask = y == 0

    feature_stats = {}
    for i, col in enumerate(feature_cols):
        feature_stats[col] = {
            'mean_legit': float(np.mean(X[legit_mask, i])),
            'std_legit': float(np.std(X[legit_mask, i])),
            'mean_fraud': float(np.mean(X[fraud_mask, i])),
            'std_fraud': float(np.std(X[fraud_mask, i])),
            'overall_mean': float(np.mean(X[:, i])),
            'overall_std': float(np.std(X[:, i])),
            'correlation_with_fraud': float(np.corrcoef(X[:, i], y)[0, 1])
        }
    return feature_stats


def compute_feature_importance(feature_stats, feature_cols=FEATURE_COLS):
    """Pearson-correlation-based feature ranking for the frontend dashboard."""
    feature_importance = []
    for col in feature_cols:
        corr = feature_stats[col]['correlation_with_fraud']
        feature_importance.append({
            'feature': col,
            'correlation': round(corr, 6),
            'abs_correlation': round(abs(corr), 6)
        })
    feature_importance.sort(key=lambda x: x['abs_correlation'], reverse=True)
    return feature_importance
