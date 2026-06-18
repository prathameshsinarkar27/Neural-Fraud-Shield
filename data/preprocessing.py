"""
data/preprocessing.py
======================
Handles train/test split, StandardScaler creation, SMOTE oversampling,
and persistence of the fitted scaler.

Migrated from: train_model.py  (STEPS 3 & 4)
"""

import os
import json
import pickle
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def split_and_scale(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train/test split and fit a StandardScaler on training data.

    Parameters
    ----------
    X            : np.ndarray  — feature matrix
    y            : np.ndarray  — label vector
    test_size    : float       — fraction reserved for testing (default 0.2)
    random_state : int

    Returns
    -------
    tuple
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Train fraud ratio: {y_train.mean() * 100:.4f}%")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(X_train_scaled: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Apply SMOTE oversampling to the scaled training set to handle class imbalance.

    Parameters
    ----------
    X_train_scaled : np.ndarray
    y_train        : np.ndarray
    random_state   : int

    Returns
    -------
    tuple
        X_train_smote, y_train_smote, smote_data
        smote_data is a dict with 'before' and 'after' class counts (JSON-serialisable).
    """
    smote_before = {str(k): int(v) for k, v in Counter(y_train).items()}
    print(f"Before SMOTE: {Counter(y_train)}")

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    smote_after = {str(k): int(v) for k, v in Counter(y_train_smote).items()}
    print(f"After SMOTE:  {Counter(y_train_smote)}")

    smote_data = {"before": smote_before, "after": smote_after}
    return X_train_smote, y_train_smote, smote_data


def save_preprocessing_artifacts(
    scaler: StandardScaler,
    smote_data: dict,
    artifacts_dir: str = "artifacts",
) -> None:
    """
    Persist scaler.pkl and smote_data.json.

    Parameters
    ----------
    scaler       : StandardScaler — fitted scaler
    smote_data   : dict           — class counts before/after SMOTE
    artifacts_dir : str
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved to artifacts/scaler.pkl")

    smote_path = os.path.join(artifacts_dir, "smote_data.json")
    with open(smote_path, "w") as f:
        json.dump(smote_data, f, indent=2)
    print("SMOTE data saved to artifacts/smote_data.json")
