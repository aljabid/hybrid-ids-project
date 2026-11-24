import os
import sys
import json
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.autoencoder.autoencoder_model import Autoencoder
from utils.feature_config import FEATURE_NAMES
from utils.logger import get_logger
import yaml
logger = get_logger(__name__)
VAL_PATH     = "data/splits/local/X_val_normal_only.csv"
WEIGHTS_PATH = "models/autoencoder/saved/autoencoder_weights.pth"
THRESHOLD_OUT = "models/autoencoder/saved/threshold.json"
def compute_threshold(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ae_cfg      = config['autoencoder']
    hidden_dims = ae_cfg['hidden_dims']
    _state = torch.load(WEIGHTS_PATH, map_location='cpu')
    input_dim = _state['encoder.0.weight'].shape[1]
    logger.info(f"Inferred input_dim from weights: {input_dim}")
    model = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    logger.info(f"Autoencoder loaded from {WEIGHTS_PATH}")
    logger.info(f"Loading validation set: {VAL_PATH}")
    X_val = pd.read_csv(VAL_PATH)
    meta_cols = ['label', 'label_str']
    available = [c for c in X_val.columns if c not in meta_cols]
    X_val_tensor = torch.FloatTensor(X_val[available].values)
    with torch.no_grad():
        errors = model.reconstruction_error(X_val_tensor).numpy()
    mu    = float(np.mean(errors))
    sigma = float(np.std(errors))
    theta = mu + 2 * sigma
    logger.info(f"Reconstruction errors on normal validation set:")
    logger.info(f"  Mean (μ):    {mu:.6f}")
    logger.info(f"  Std  (σ):    {sigma:.6f}")
    logger.info(f"  Threshold θ: {theta:.6f}  (μ + 2σ)")
    logger.info(f"  Min error:   {errors.min():.6f}")
    logger.info(f"  Max error:   {errors.max():.6f}")
    result = {
        "threshold":    round(float(theta), 6),
        "mu":           round(float(mu), 6),
        "sigma":        round(float(sigma), 6),
        "method":       "mean_plus_2std",
        "n_val_flows":  len(X_val),
        "error_min":    round(float(errors.min()), 6),
        "error_max":    round(float(errors.max()), 6),
        "error_p95":    round(float(np.percentile(errors, 95)), 6),
        "error_p99":    round(float(np.percentile(errors, 99)), 6),
    }
    os.makedirs(os.path.dirname(THRESHOLD_OUT), exist_ok=True)
    with open(THRESHOLD_OUT, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Threshold saved → {THRESHOLD_OUT}")
    return result
if __name__ == "__main__":
    compute_threshold()
