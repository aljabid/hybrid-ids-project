   
                         
                                                      
                                                         
                                                 
                                                      

                                                           
                                                              
   

import os
import sys
import json
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.logger import get_logger

logger = get_logger(__name__)

INPUT_PATH  = "data/processed/cic_ids2018/cic2018_raw_selected.csv"
OUTPUT_PATH = "data/processed/cic_ids2018/cic2018_balanced_subset.csv"
STATS_OUT   = "results/metrics/cic2018_subset_stats.json"


def create_subset(input_path=INPUT_PATH, output_path=OUTPUT_PATH,
                  config_path="config/config.yaml"):

    with open(config_path) as f:
        config = yaml.safe_load(f)

                   
    n_attack   = config['cic_ids2018']['n_attack_samples']              
    ratio      = config['cic_ids2018']['benign_to_attack_ratio']          
    seed       = config['cic_ids2018']['random_seed']
    n_benign   = int(n_attack * ratio)

    logger.info(f"Subset config — FTP: {n_attack} | SSH: {n_attack} | Benign: {n_benign} | Seed: {seed}")

    df = pd.read_csv(input_path)

    ftp_df    = df[df['Label'] == 1]
    ssh_df    = df[df['Label'] == 2]
    benign_df = df[df['Label'] == 0]

    ftp_sample    = ftp_df.sample(n=min(n_attack, len(ftp_df)), random_state=seed)
    ssh_sample    = ssh_df.sample(n=min(n_attack, len(ssh_df)), random_state=seed)
    benign_sample = benign_df.sample(n=min(n_benign, len(benign_df)), random_state=seed)

    subset = pd.concat([ftp_sample, ssh_sample, benign_sample])\
               .sample(frac=1, random_state=seed)\
               .reset_index(drop=True)

    logger.info(f"Final subset: {len(subset)} rows")
    logger.info(f"Distribution:\n{subset['Label'].value_counts().to_string()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subset.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")

                                  
    stats = {
        "total_rows": int(len(subset)),
        "n_ftp":      int((subset['Label'] == 1).sum()),
        "n_ssh":      int((subset['Label'] == 2).sum()),
        "n_benign":   int((subset['Label'] == 0).sum()),
        "ratio":      ratio,
        "random_seed": seed,
        "source_dataset": "CIC-IDS2018",
        "filter": "Benign + FTP + SSH"
    }
    os.makedirs(os.path.dirname(STATS_OUT), exist_ok=True)
    with open(STATS_OUT, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Subset stats saved → {STATS_OUT}")

    return subset, stats



if __name__ == "__main__":
    create_subset()
