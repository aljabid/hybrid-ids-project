import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger
logger = get_logger(__name__)
INPUT_PATH  = "data/processed/cic_ids2018/cic2018_raw_selected.csv"
OUTPUT_PATH = "data/processed/cic_ids2018/cic2018_aligned_features.csv"
CICIDS2018_RAW_MAP = {
    'TotLen Fwd Pkts': '_byte_count_raw',
    'Tot Fwd Pkts':    '_packet_count_raw',
    'Flow Pkts/s':     'packet_rate',
    'ACK Flag Cnt':    'ack_flag_count',
    'SYN Flag Cnt':    'syn_flag_count',
    'FIN Flag Cnt':    'fin_flag_count',
}
MI_SELECTED = [
    'byte_count', 'bytes_per_packet', 'log_byte_count',
    'ack_flag_count', 'flag_sum', 'ack_ratio',
    'fin_ratio', 'syn_ratio', 'packet_rate', 'log_packet_rate'
]
def align(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = df.columns.str.strip()
    if 'Label' in df.columns:
        df.rename(columns={'Label': 'label'}, inplace=True)
    logger.info("Renaming CIC-IDS2018 raw columns")
    df = df.rename(columns=CICIDS2018_RAW_MAP)
    eps = 1e-9
    df['byte_count']       = df['_byte_count_raw']
    df['bytes_per_packet'] = df['_byte_count_raw'] / (df['_packet_count_raw'] + eps)
    df['log_byte_count']   = np.log1p(df['_byte_count_raw'])
    df['log_packet_rate']  = np.log1p(df['packet_rate'])
    df['ack_ratio']        = df['ack_flag_count'] / (df['_packet_count_raw'] + eps)
    df['syn_ratio']        = df['syn_flag_count']  / (df['_packet_count_raw'] + eps)
    df['fin_ratio']        = df['fin_flag_count']  / (df['_packet_count_raw'] + eps)
    df['flag_sum']         = (df['syn_flag_count'] +
                               df['ack_flag_count'] +
                               df['fin_flag_count'])
    missing = [f for f in MI_SELECTED if f not in df.columns]
    if missing:
        logger.error(f"Missing features after derivation: {missing}")
        raise ValueError(f"Cannot align: {missing}")
    df_aligned = df[MI_SELECTED + ['label']].copy()
    df_aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_aligned.dropna(inplace=True)
    df_aligned.reset_index(drop=True, inplace=True)
    logger.info(f"Aligned rows: {len(df_aligned)}")
    logger.info(f"Features: {MI_SELECTED}")
    logger.info(f"Label distribution:\n{df_aligned['label'].value_counts().to_string()}")
    for col in ['byte_count', 'log_byte_count', 'packet_rate', 'log_packet_rate']:
        if col in df_aligned.columns:
            logger.info(f"  {col}: min={df_aligned[col].min():.4f} "
                       f"max={df_aligned[col].max():.4f} "
                       f"mean={df_aligned[col].mean():.4f}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_aligned.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df_aligned
if __name__ == "__main__":
    align()
