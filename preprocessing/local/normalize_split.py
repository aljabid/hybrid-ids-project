   
                  
                                                                       
           
                                                                     
                                                                           

                                                  
                                     
                                    
                                               
                                     
                                    
                                          
   

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.feature_config import FEATURE_NAMES
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)

INPUT_PATH   = "data/processed/local/flows_mi_selected.csv"
SPLITS_DIR   = "data/splits/local"
SCALER_PATH  = "models/classifier/saved/scaler.pkl"


def run(input_path=INPUT_PATH, config_path="config/config.yaml"):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    test_size   = config['classifier']['test_size']
    random_seed = config['classifier']['random_state']

    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)

                                                                        
    meta_cols = ['label', 'label_str']
    available_features = [c for c in df.columns if c not in meta_cols]
    logger.info(f"Using {len(available_features)} features for split")

    X = df[available_features].values
    y = df['label'].values

                                                              
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

                                                              
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)                      
    logger.info("Min-Max scaler fitted on training data")

                                                              
    normal_mask      = (y_train == 0)
    X_val_normal     = X_train_scaled[normal_mask]
    logger.info(f"Normal-only validation set: {len(X_val_normal)} flows")

                                                              
    os.makedirs(SPLITS_DIR, exist_ok=True)

    pd.DataFrame(X_train_scaled,
                 columns=available_features).to_csv(
        os.path.join(SPLITS_DIR, "X_train.csv"), index=False)

    pd.DataFrame(X_test_scaled,
                 columns=available_features).to_csv(
        os.path.join(SPLITS_DIR, "X_test.csv"), index=False)

    pd.DataFrame(X_val_normal,
                 columns=available_features).to_csv(
        os.path.join(SPLITS_DIR, "X_val_normal_only.csv"), index=False)

    pd.DataFrame(y_train, columns=['label']).to_csv(
        os.path.join(SPLITS_DIR, "y_train.csv"), index=False)

    pd.DataFrame(y_test, columns=['label']).to_csv(
        os.path.join(SPLITS_DIR, "y_test.csv"), index=False)

                                                               
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"Scaler saved → {SCALER_PATH}")
    logger.info(f"All splits saved → {SPLITS_DIR}/")

                                                              
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Train class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    unique, counts = np.unique(y_test, return_counts=True)
    logger.info(f"Test  class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return {
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "n_val_normal": len(X_val_normal),
        "features": available_features
    }


if __name__ == "__main__":
    run()
