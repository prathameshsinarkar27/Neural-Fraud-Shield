"""
data/feature_engineering.py
============================
Computes engineered features, feature statistics, and feature importance.

Migrated from: train_model.py  (STEP 2)
"""

import os
import json
import numpy as np
import pandas as pd


FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Log_Amount", "Hour"]


def engineer_features(df: pd.DataFrame):
    """
    Add Log_Amount and Hour columns, then extract the feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from loader.load_dataset().

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        X  — feature matrix  (n_samples, 30)
        y  — label vector    (n_samples,)
        feature_cols — ordered list of feature names
    """
    df = df.copy()
    df["Log_Amount"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] % 86400) / 3600  # seconds → hour of day

    feature_cols = FEATURE_COLS
    X = df[feature_cols].values
    y = df["Class"].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features used: {feature_cols}")
    return X, y, feature_cols


def compute_feature_stats(X: np.ndarray, y: np.ndarray, feature_cols: list[str]) -> dict:
    """
    Compute per-feature statistics split by class (legit vs fraud).

    Used by the translation layer at inference time.

    Parameters
    ----------
    X : np.ndarray  — feature matrix
    y : np.ndarray  — label vector
    feature_cols : list[str]

    Returns
    -------
    dict  — {feature_name: {mean_legit, std_legit, mean_fraud, std_fraud,
                             overall_mean, overall_std, correlation_with_fraud}}
    """
    fraud_mask = y == 1
    legit_mask = y == 0

    feature_stats = {}
    for i, col in enumerate(feature_cols):
        feature_stats[col] = {
            "mean_legit":            float(np.mean(X[legit_mask, i])),
            "std_legit":             float(np.std(X[legit_mask, i])),
            "mean_fraud":            float(np.mean(X[fraud_mask, i])),
            "std_fraud":             float(np.std(X[fraud_mask, i])),
            "overall_mean":          float(np.mean(X[:, i])),
            "overall_std":           float(np.std(X[:, i])),
            "correlation_with_fraud": float(np.corrcoef(X[:, i], y)[0, 1]),
        }
    return feature_stats


def compute_feature_importance(feature_stats: dict, feature_cols: list[str]) -> list[dict]:
    """
    Rank features by absolute Pearson correlation with the fraud label.

    Parameters
    ----------
    feature_stats : dict  — output of compute_feature_stats()
    feature_cols  : list[str]

    Returns
    -------
    list[dict]  — sorted by abs_correlation descending
    """
    importance = []
    for col in feature_cols:
        corr = feature_stats[col]["correlation_with_fraud"]
        importance.append({
            "feature":         col,
            "correlation":     round(corr, 6),
            "abs_correlation": round(abs(corr), 6),
        })
    importance.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return importance


def save_feature_artifacts(
    feature_stats: dict,
    feature_importance: list[dict],
    feature_cols: list[str],
    artifacts_dir: str = "artifacts",
) -> None:
    """
    Persist feature_stats.json, feature_importance.json, and feature_cols.json.

    Parameters
    ----------
    feature_stats     : dict
    feature_importance : list[dict]
    feature_cols      : list[str]
    artifacts_dir     : str
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(os.path.join(artifacts_dir, "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)
    print("Feature statistics saved to artifacts/feature_stats.json")

    with open(os.path.join(artifacts_dir, "feature_importance.json"), "w") as f:
        json.dump(feature_importance, f, indent=2)
    print("Feature importance saved to artifacts/feature_importance.json")

    with open(os.path.join(artifacts_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print("Feature columns saved to artifacts/feature_cols.json")
