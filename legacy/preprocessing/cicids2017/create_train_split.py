import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger
logger = get_logger(__name__)
INPUT_PATH  = "data/processed/cicids2017/cicids_aligned_features.csv"
SPLITS_DIR  = "data/splits/local"
SCALER_PATH = "models/classifier/saved/scaler.pkl"
def create_cicids_train_split():
    logger.info("Creating training split from CICIDS2017 real traffic data")
    df = pd.read_csv(INPUT_PATH)
    meta_cols    = ['label', 'label_str']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    n_per_class = 5000
    normal_df   = df[df['label'] == 0].sample(
                      n=min(n_per_class, (df['label']==0).sum()),
                      random_state=42)
    attack_df   = df[df['label'] == 1].sample(
                      n=min(n_per_class, (df['label']==1).sum()),
                      random_state=42)
    balanced = pd.concat([normal_df, attack_df]).sample(
                   frac=1, random_state=42).reset_index(drop=True)
    X = balanced[feature_cols].values
    y = balanced['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    normal_mask  = (y_train == 0)
    X_val_normal = X_train_scaled[normal_mask]
    os.makedirs(SPLITS_DIR, exist_ok=True)
    pd.DataFrame(X_train_scaled,
                 columns=feature_cols).to_csv(
        os.path.join(SPLITS_DIR, "X_train.csv"), index=False)
    pd.DataFrame(X_test_scaled,
                 columns=feature_cols).to_csv(
        os.path.join(SPLITS_DIR, "X_test.csv"), index=False)
    pd.DataFrame(X_val_normal,
                 columns=feature_cols).to_csv(
        os.path.join(SPLITS_DIR, "X_val_normal_only.csv"), index=False)
    pd.DataFrame(y_train, columns=['label']).to_csv(
        os.path.join(SPLITS_DIR, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=['label']).to_csv(
        os.path.join(SPLITS_DIR, "y_test.csv"), index=False)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    cicids_full = pd.read_csv(
        "data/processed/cicids2017/cicids_aligned_features.csv")
    cicids_features = [c for c in cicids_full.columns
                       if c not in meta_cols]
    X_cicids   = cicids_full[cicids_features].values
    y_cicids   = cicids_full['label'].values
    X_cicids_s = scaler.transform(X_cicids)
    cicids_splits = "data/splits/cicids2017"
    os.makedirs(cicids_splits, exist_ok=True)
    pd.DataFrame(X_cicids_s,
                 columns=cicids_features).to_csv(
        os.path.join(cicids_splits, "X_test_cicids.csv"), index=False)
    pd.DataFrame(y_cicids, columns=['label']).to_csv(
        os.path.join(cicids_splits, "y_test_cicids.csv"), index=False)
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | "
                f"Val normal: {len(X_val_normal)}")
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Scaler refitted on CICIDS2017 training data")
    logger.info(f"CICIDS2017 test set re-normalized with new scaler")
    logger.info(f"Class distribution train: "
                f"{dict(zip(*np.unique(y_train, return_counts=True)))}")
    return feature_cols
