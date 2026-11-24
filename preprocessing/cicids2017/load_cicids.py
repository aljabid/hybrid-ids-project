import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger
logger = get_logger(__name__)
INPUT_PATH  = "data/raw/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
OUTPUT_PATH = "data/processed/cicids2017/cicids_raw_selected.csv"
KEEP_LABELS = ["BENIGN", "PortScan"]
LABEL_REMAP = {"BENIGN": 0, "PortScan": 1}
def load(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    logger.info(f"Loading CICIDS2017 from {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()
    label_col = None
    for col in df.columns:
        if col.strip().lower() == 'label':
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")
    df.rename(columns={label_col: 'Label'}, inplace=True)
    df['Label'] = df['Label'].str.strip()
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Label distribution:\n{df['Label'].value_counts().to_string()}")
    df = df[df['Label'].isin(KEEP_LABELS)].copy()
    df['Label'] = df['Label'].map(LABEL_REMAP)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Label'], inplace=True)
    logger.info(f"After filtering: {len(df)} rows")
    logger.info(f"Binary label distribution:\n{df['Label'].value_counts().to_string()}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df
if __name__ == "__main__":
    load()
