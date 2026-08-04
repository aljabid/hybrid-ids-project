   
                       
                                                            
                                                           
                                                

                                              
                                                   
   

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.feature_config import FEATURE_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

INPUT_PATH  = "data/processed/local/flows_labeled.csv"
OUTPUT_PATH = "data/processed/local/flows_all_features.csv"


def extract(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)

                                         
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        logger.error(f"Missing canonical features: {missing}")
        raise ValueError(f"Missing features: {missing}")

    logger.info(f"All {len(FEATURE_NAMES)} canonical features present")

                                                        
    eps = 1e-9

    df['bytes_per_packet']    = df['byte_count'] / (df['packet_count'] + eps)
    df['syn_ratio']           = df['syn_flag_count'] / (df['packet_count'] + eps)
    df['ack_ratio']           = df['ack_flag_count'] / (df['packet_count'] + eps)
    df['fin_ratio']           = df['fin_flag_count'] / (df['packet_count'] + eps)
    df['flag_sum']            = (df['syn_flag_count'] +
                                  df['ack_flag_count'] +
                                  df['fin_flag_count'])
    df['duration_per_packet'] = df['flow_duration'] / (df['packet_count'] + eps)
    df['log_packet_rate']     = np.log1p(df['packet_rate'])
    df['log_byte_count']      = np.log1p(df['byte_count'])

    all_features = FEATURE_NAMES + [
        'bytes_per_packet', 'syn_ratio', 'ack_ratio', 'fin_ratio',
        'flag_sum', 'duration_per_packet', 'log_packet_rate', 'log_byte_count'
    ]

    logger.info(f"Total features available for MI ranking: {len(all_features)}")

    out_df = df[all_features + ['label', 'label_str']].copy()
    out_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    out_df.dropna(inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(out_df)} rows, "
                f"{len(all_features)} features → {output_path}")

    return out_df, all_features


if __name__ == "__main__":
    extract()
