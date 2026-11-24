import os
import sys
import pickle
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.feature_config import FEATURE_NAMES
from utils.logger import get_logger
logger = get_logger(__name__)
INPUT_PATH  = "data/processed/cic_ids2018/cic2018_aligned_features.csv"
SCALER_PATH = "models/classifier/saved/scaler.pkl"
NORM_OUT    = "data/processed/cic_ids2018/cic2018_normalized.csv"
SPLITS_DIR  = "data/splits/cic_ids2018"
def normalize(input_path=INPUT_PATH, scaler_path=SCALER_PATH):
    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Applying LOCAL scaler — transform() only, no refitting")
    meta_cols = ['label', 'label_str']
    available_features = [c for c in df.columns if c not in meta_cols]
    meta_cols = ['label', 'label_str']
    available_features = [c for c in df.columns if c not in meta_cols]
    X_scaled = scaler.transform(df[available_features].values)
    df_out = pd.DataFrame(X_scaled, columns=available_features)
    df_out['label'] = df['label'].values
    os.makedirs(os.path.dirname(NORM_OUT), exist_ok=True)
    df_out.to_csv(NORM_OUT, index=False)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    df_out[available_features].to_csv(
        os.path.join(SPLITS_DIR, "X_test_cic2018.csv"), index=False)
    df_out[['label']].to_csv(
        os.path.join(SPLITS_DIR, "y_test_cic2018.csv"), index=False)
    logger.info(f"CIC-IDS2018 normalized: {len(df_out)} rows")
    logger.info(f"Splits saved → {SPLITS_DIR}/")
    return df_out
if __name__ == "__main__":
    normalize()
