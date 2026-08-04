   
                       
                                        
                                                    
                                                 

                                                
                                                     
                                                 
   

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.autoencoder.autoencoder_model import Autoencoder
from utils.feature_config import FEATURE_NAMES
from utils.metrics_utils import compute_tpr_fpr
from utils.plot_utils import plot_reconstruction_error
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)

X_TEST_PATH   = "data/splits/local/X_test.csv"
Y_TEST_PATH   = "data/splits/local/y_test.csv"
WEIGHTS_PATH  = "models/autoencoder/saved/autoencoder_weights.pth"
THRESHOLD_PATH = "models/autoencoder/saved/threshold.json"
METRICS_OUT   = "results/metrics/autoencoder_metrics.json"
FIGURE_OUT    = "results/figures/reconstruction_error_plot.png"


def run(config_path="config/config.yaml"):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ae_cfg = config['autoencoder']

                              
    hidden_dims = ae_cfg['hidden_dims']

                                        
    _state = torch.load(WEIGHTS_PATH, map_location='cpu')
    input_dim = _state['encoder.0.weight'].shape[1]

    model = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(_state)
    model.eval()

    logger.info(f"Inferred input_dim from weights: {input_dim}")

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.eval()

    with open(THRESHOLD_PATH) as f:
        thresh_data = json.load(f)
    threshold = thresh_data['threshold']
    logger.info(f"Threshold: {threshold:.6f}")

                    
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    meta_cols = ['label', 'label_str']
    available = [c for c in X_test.columns if c not in meta_cols]
    X_tensor = torch.FloatTensor(X_test[available].values)

                                   
    with torch.no_grad():
        errors = model.reconstruction_error(X_tensor).numpy()

                                           
    y_binary  = (y_test != 0).astype(int)
    y_pred_ae = (errors > threshold).astype(int)

                              
    normal_errors = errors[y_binary == 0]
    attack_errors = errors[y_binary == 1]

             
    tpr, fpr = compute_tpr_fpr(y_binary, y_pred_ae)
    fpr_curve, tpr_curve, _ = roc_curve(y_binary, errors)
    auc_score = float(auc(fpr_curve, tpr_curve))

    metrics = {
        "threshold":          round(threshold, 6),
        "n_normal":           int(len(normal_errors)),
        "n_attack":           int(len(attack_errors)),
        "mean_error_normal":  round(float(normal_errors.mean()), 6),
        "mean_error_attack":  round(float(attack_errors.mean()), 6),
        "std_error_normal":   round(float(normal_errors.std()), 6),
        "std_error_attack":   round(float(attack_errors.std()), 6),
        "tpr":                round(tpr, 4),
        "fpr":                round(fpr, 4),
        "auc":                round(auc_score, 4),
        "n_flagged":          int(y_pred_ae.sum()),
        "detection_rate":     round(float(y_pred_ae[y_binary==1].mean()), 4),
        "false_alarm_rate":   round(float(y_pred_ae[y_binary==0].mean()), 4),
    }

    logger.info(f"Autoencoder evaluation:")
    logger.info(f"  TPR:  {metrics['tpr']:.4f}")
    logger.info(f"  FPR:  {metrics['fpr']:.4f}")
    logger.info(f"  AUC:  {metrics['auc']:.4f}")
    logger.info(f"  Mean error (normal): {metrics['mean_error_normal']:.6f}")
    logger.info(f"  Mean error (attack): {metrics['mean_error_attack']:.6f}")

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {METRICS_OUT}")

    os.makedirs(os.path.dirname(FIGURE_OUT), exist_ok=True)
    plot_reconstruction_error(normal_errors, attack_errors,
                               threshold, FIGURE_OUT)

    return metrics


if __name__ == "__main__":
    run()
