"""
data/loader.py
==============
Handles dataset loading and optional Kaggle auto-download.

Migrated from: train_model.py  (STEP 1)
"""

import os
import glob
import shutil
import pandas as pd


DATASET_PATH = "creditcard.csv"


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the Credit Card Fraud Detection dataset.

    If the CSV is not present locally, attempts an automatic download
    via kagglehub.  Raises SystemExit with clear instructions on failure.

    Parameters
    ----------
    path : str
        Local path to creditcard.csv (default: project root).

    Returns
    -------
    pd.DataFrame
        Raw dataset with shape (~284807, 31).
    """
    if not os.path.exists(path):
        print("Dataset not found locally. Attempting download via kagglehub...")
        try:
            import kagglehub
            downloaded_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            csv_files = glob.glob(
                os.path.join(downloaded_path, "**", "creditcard.csv"), recursive=True
            )
            if csv_files:
                shutil.copy(csv_files[0], path)
                print(f"Dataset copied from {csv_files[0]}")
            else:
                raise FileNotFoundError("creditcard.csv not found in downloaded path")
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print("Please download creditcard.csv from:")
            print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print(f"and place it at: {path}")
            raise SystemExit(1)

    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    print(f"Fraud ratio: {df['Class'].mean() * 100:.4f}%")
    return df
