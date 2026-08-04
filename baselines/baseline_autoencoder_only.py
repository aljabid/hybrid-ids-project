   
                            
                                             
                                                    
   

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

                                                                                      

from models.autoencoder.autoencoder_model import Autoencoder
from utils.feature_config import FEATURE_NAMES
from utils.metrics_utils import full_metrics_dict
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)

AE_WEIGHTS     = "models/autoencoder/saved/autoencoder_weights.pth"
THRESHOLD_PATH = "models/autoencoder/saved/threshold.json"
X_TEST_PATH    = "data/splits/local/X_test.csv"
Y_TEST_PATH    = "data/splits/local/y_test.csv"
METRICS_OUT    = "results/metrics/baseline_autoencoder_only.json"


def run(config_path="config/config.yaml"):
                                                                                                               

    with open(config_path) as f:
        config = yaml.safe_load(f)


    ae_cfg = config['autoencoder']

                                                     
    _state = torch.load(AE_WEIGHTS, map_location='cpu')
    input_dim = _state['encoder.0.weight'].shape[1]

    ae = Autoencoder(input_dim=input_dim,
                 hidden_dims=ae_cfg['hidden_dims'])
    ae.load_state_dict(_state)
    ae.eval()

    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)['threshold']

    X = pd.read_csv(X_TEST_PATH)
    y = pd.read_csv(Y_TEST_PATH).values.ravel()


    meta_cols = ['label', 'label_str']
    available = [c for c in X.columns if c not in meta_cols]
    X_tensor = torch.FloatTensor(X[available].values)

    with torch.no_grad():
        errors = ae.reconstruction_error(X_tensor).numpy()

    y_binary   = (y != 0).astype(int)
    pred_binary = (errors > threshold).astype(int)

    metrics = full_metrics_dict(y_binary, pred_binary,
                                dataset_name="local_autoencoder_only")
    metrics['model']     = "Autoencoder Only"
    metrics['threshold'] = round(threshold, 6)

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Baseline (Autoencoder Only): "
                f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | "
                f"TPR={metrics['tpr']:.4f} | FPR={metrics['fpr']:.4f}")
    return metrics


if __name__ == "__main__":
    run()
