import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.feature_config import FEATURE_NAMES
from utils.metrics_utils import full_metrics_dict, compute_roc_auc
from utils.logger import get_logger
logger = get_logger(__name__)
MODEL_PATH   = "models/classifier/saved/decision_tree_model.pkl"
X_TEST_PATH  = "data/splits/local/X_test.csv"
Y_TEST_PATH  = "data/splits/local/y_test.csv"
METRICS_OUT  = "results/metrics/baseline_classifier_only.json"
def run():
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    X = pd.read_csv(X_TEST_PATH)
    y = pd.read_csv(Y_TEST_PATH).values.ravel()
    meta_cols = ['label', 'label_str']
    available = [c for c in X.columns if c not in meta_cols]
    X = X[available].values
    y_pred = clf.predict(X)
    proba  = clf.predict_proba(X)
    y_binary   = (y != 0).astype(int)
    pred_binary = (y_pred != 0).astype(int)
    metrics = full_metrics_dict(y_binary, pred_binary,
                                dataset_name="local_classifier_only")
    try:
        metrics['auc'] = round(compute_roc_auc(y_binary, 1-proba[:, 0]), 4)
    except Exception:
        metrics['auc'] = 0.0
    metrics['model'] = "Decision Tree Only"
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Baseline (Classifier Only): "
                f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | "
                f"TPR={metrics['tpr']:.4f} | FPR={metrics['fpr']:.4f}")
    return metrics
if __name__ == "__main__":
    run()
