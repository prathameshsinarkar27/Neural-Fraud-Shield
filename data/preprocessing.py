"""
data/preprocessing.py
==============================================================================
Data loading and preprocessing for the Neural Fraud Shield.
  - Loading creditcard.csv (auto-downloading via kagglehub if missing)
  - Splitting into train/test and standard-scaling the features
  - Applying SMOTE to balance the extreme class imbalance (0.17% fraud)
==============================================================================
"""

import os
import glob
import shutil
import pickle
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

DATASET_PATH = "data/creditcard.csv"


def load_dataset(dataset_path=DATASET_PATH):
    """Load creditcard.csv, auto-downloading via kagglehub if not present locally."""
    if not os.path.exists(dataset_path):
        print("Dataset not found locally. Attempting download via kagglehub...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            csv_files = glob.glob(os.path.join(path, "**", "creditcard.csv"), recursive=True)
            if csv_files:
                shutil.copy(csv_files[0], dataset_path)
                print(f"Dataset copied from {csv_files[0]}")
            else:
                raise FileNotFoundError("CSV not found in downloaded path")
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print("Please download creditcard.csv from:")
            print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print("and place it in the data/ directory as data/creditcard.csv.")
            exit(1)

    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    print(f"Fraud ratio: {df['Class'].mean()*100:.4f}%")
    return df


def split_and_scale(X, y, artifacts_dir="artifacts"):
    """Stratified train/test split followed by StandardScaler fit on train only.

    Saves the fitted scaler to <artifacts_dir>/scaler.pkl, exactly as before.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Train fraud ratio: {y_train.mean()*100:.4f}%")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(os.path.join(artifacts_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(X_train_scaled, y_train):
    """Apply SMOTE oversampling and return the before/after class counts."""
    print(f"Before SMOTE: {Counter(y_train)}")
    smote_before = dict(Counter(y_train))

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"After SMOTE:  {Counter(y_train_smote)}")
    smote_after = dict(Counter(y_train_smote))

    smote_data = {
        'before': {str(k): int(v) for k, v in smote_before.items()},
        'after': {str(k): int(v) for k, v in smote_after.items()}
    }
    return X_train_smote, y_train_smote, smote_data
