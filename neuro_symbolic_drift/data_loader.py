"""
data_loader.py
--------------
Load and split the Kaggle Credit Card Fraud dataset.

Protocol (same as Articles 1 & 2 for direct comparability):
    70 / 15 / 15 stratified split
    StandardScaler fit on train only
    pos_weight = count(y=0) / count(y=1) ≈ 578
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from torch.utils.data        import DataLoader, TensorDataset

DATA_PATH     = Path("data/creditcard.csv")
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
TARGET        = "Class"


def load_data(seed: int = 42, val_size=0.15, test_size=0.15):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {DATA_PATH}\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place creditcard.csv in the data/ directory."
        )
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    y  = df[TARGET].values.astype(np.float32)

    X_tr_val, X_test, y_tr_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    adj_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_val, y_tr_val, test_size=adj_val,
        stratify=y_tr_val, random_state=seed
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler, FEATURE_NAMES, pos_weight)


def get_dataloaders(X_train, y_train, X_val, y_val, batch_size=2048):
    def _loader(X, y, shuffle):
        ds = TensorDataset(torch.FloatTensor(X),
                           torch.FloatTensor(y).unsqueeze(1))
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, drop_last=False)
    return _loader(X_train, y_train, True), _loader(X_val, y_val, False)


def print_dataset_stats(y_train, y_val, y_test):
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        n_fraud = int(y.sum())
        n       = len(y)
        print(f"  {name:5s}: {n:7d} samples | "
              f"{n_fraud:4d} fraud ({100*n_fraud/n:.3f}%)")
