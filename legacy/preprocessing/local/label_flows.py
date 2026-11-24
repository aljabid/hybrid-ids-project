import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.feature_config import LABEL_MAP_LOCAL
from utils.logger import get_logger
logger = get_logger(__name__)
INPUT_PATH  = "data/processed/local/flows_cleaned.csv"
OUTPUT_PATH = "data/processed/local/flows_labeled.csv"
def label(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Applying label mapping: {LABEL_MAP_LOCAL}")
    unmapped = set(df['label'].unique()) - set(LABEL_MAP_LOCAL.keys())
    if unmapped:
        logger.warning(f"Unmapped label values found: {unmapped} — dropping")
        df = df[df['label'].isin(LABEL_MAP_LOCAL.keys())]
    df['label_str']  = df['label']
    df['label']      = df['label'].map(LABEL_MAP_LOCAL)
    logger.info(f"Label distribution (integer):\n"
                f"{df['label'].value_counts().sort_index().to_string()}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df
if __name__ == "__main__":
    label()
