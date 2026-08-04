   
                                  
                                                       
                                                                   
                                                   
   

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.autoencoder.autoencoder_model import Autoencoder
from utils.feature_config import FEATURE_NAMES, LABEL_MAP_LOCAL_INV
from utils.metrics_utils import full_metrics_dict
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)

CLF_PATH       = "models/classifier/saved/decision_tree_model.pkl"
AE_WEIGHTS     = "models/autoencoder/saved/autoencoder_weights.pth"
THRESHOLD_PATH = "models/autoencoder/saved/threshold.json"
X_TEST_PATH    = "data/splits/local/X_test.csv"
Y_TEST_PATH    = "data/splits/local/y_test.csv"
METRICS_OUT    = "results/metrics/baseline_fixed_threshold_hybrid.json"


def run(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ae_cfg = config['autoencoder']

    with open(CLF_PATH, 'rb') as f:
        clf = pickle.load(f)

    _state = torch.load(AE_WEIGHTS, map_location='cpu')
    input_dim = _state['encoder.0.weight'].shape[1]

    ae = Autoencoder(input_dim=input_dim,
                     hidden_dims=ae_cfg['hidden_dims'])
    ae.load_state_dict(_state)
    ae.eval()

    with open(THRESHOLD_PATH) as f:
        ae_threshold = json.load(f)['threshold']

    X = pd.read_csv(X_TEST_PATH)
    y = pd.read_csv(Y_TEST_PATH).values.ravel()
    meta_cols = ['label', 'label_str']
    available = [c for c in X.columns if c not in meta_cols]
    X_arr = X[available].values

    y_pred_clf   = clf.predict(X_arr)
    X_tensor     = torch.FloatTensor(X_arr)
    with torch.no_grad():
        ae_errors = ae.reconstruction_error(X_tensor).numpy()

    predictions = []
    for i in range(len(X_arr)):
        clf_label = LABEL_MAP_LOCAL_INV.get(int(y_pred_clf[i]), "Unknown")
        if clf_label != "Normal":
                                                                     
            predictions.append(1)
        else:
                                                       
            predictions.append(1 if ae_errors[i] > ae_threshold else 0)

    y_binary    = (y != 0).astype(int)
    pred_binary = np.array(predictions)

    metrics = full_metrics_dict(y_binary, pred_binary,
                                dataset_name="local_fixed_hybrid")
    metrics['model'] = "Fixed-Threshold Hybrid"

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Baseline (Fixed Hybrid): "
                f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | "
                f"TPR={metrics['tpr']:.4f} | FPR={metrics['fpr']:.4f}")
    return metrics


if __name__ == "__main__":
    run()
