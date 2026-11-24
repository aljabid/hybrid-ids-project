import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger
logger = get_logger(__name__)
INPUT_PATH  = "data/processed/local/flows_raw.csv"
OUTPUT_PATH = "data/processed/local/flows_cleaned.csv"
MIN_PACKETS = 3
def clean(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    n_start = len(df)
    logger.info(f"Rows loaded: {n_start}")
    before = len(df)
    df = df[df['packet_count'] >= MIN_PACKETS]
    logger.info(f"After min_packets filter ({MIN_PACKETS}): "
                f"{len(df)} rows  ({before - len(df)} dropped)")
    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logger.info(f"After null/inf removal: "
                f"{len(df)} rows  ({before - len(df)} dropped)")
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"After deduplication: "
                f"{len(df)} rows  ({before - len(df)} dropped)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    logger.info(f"\nTotal dropped: {n_start - len(df)} "
                f"({(n_start - len(df))/n_start*100:.1f}%)")
    logger.info(f"Final row count: {len(df)}")
    logger.info(f"Class distribution:\n"
                f"{df['label'].value_counts().to_string()}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df
if __name__ == "__main__":
    clean()
