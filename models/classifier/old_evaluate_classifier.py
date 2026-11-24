import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.feature_config import FEATURE_NAMES, CLASS_NAMES_LOCAL, CLASS_NAMES_BINARY
from utils.metrics_utils import (full_metrics_dict, compute_classification_report,
                                   compute_roc_auc)
from utils.plot_utils import plot_confusion_matrix
from utils.dataset_utils import DATASET_REGISTRY
from utils.logger import get_logger
logger = get_logger(__name__)
MODEL_PATH = "models/classifier/saved/decision_tree_model.pkl"
def evaluate_on_dataset(clf, dataset_name):
    entry = DATASET_REGISTRY[dataset_name]
    X = pd.read_csv(entry['X_test'])
    y = pd.read_csv(entry['y_test']).values.ravel()
    meta_cols = ['label', 'label_str']
    available = [c for c in X.columns if c not in meta_cols]
    X = X[available].values
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)
    metrics = full_metrics_dict(y, y_pred, dataset_name=dataset_name)
    from sklearn.metrics import classification_report
    if dataset_name == 'cicids2017':
        labels = [0, 1]
        target_names = ['BENIGN', 'PortScan']
    elif dataset_name == 'local':
        labels = [0, 1, 2, 3]
        target_names = ['Normal', 'SYN Scan', 'TCP Connect', 'Xmas Scan']
    elif dataset_name == 'cic_ids2018':
        labels = [0, 1]
        target_names = ['BENIGN', 'Attack']
    else:
        labels = np.unique(y).tolist()
        target_names = [str(c) for c in labels]
    report = classification_report(
        y_true=y,
        y_pred=y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
    metrics['classification_report'] = report
    try:
        if entry['binary']:
            conf = proba[:, 1]
            metrics['auc'] = round(compute_roc_auc(y, conf), 4)
        else:
            metrics['auc'] = round(compute_roc_auc(y, proba, multi_class=True), 4)
    except Exception as e:
        logger.warning(f"AUC error: {e}")
        metrics['auc'] = 0.0
    out_path = entry['metrics_out'].replace('hybrid_', 'classifier_')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"{dataset_name} metrics saved → {out_path}")
    plot_confusion_matrix(y, y_pred, labels,
                          title=f"Confusion Matrix — {dataset_name}",
                          output_path=entry['cm_out'])
    logger.info(f"{dataset_name} | Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f}")
    return metrics
def run():
    logger.info("Loading classifier")
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    results = {}
    logger.info("\n── Evaluating on LOCAL test set ──")
    results['local'] = evaluate_on_dataset(clf, 'local')
    cicids_path = DATASET_REGISTRY['cicids2017']['X_test']
    if os.path.exists(cicids_path):
        logger.info("\n── Evaluating on CICIDS2017 ──")
        results['cicids2017'] = evaluate_on_dataset(clf, 'cicids2017')
    else:
        logger.warning("CICIDS2017 splits not found — skipping")
    cic2018_path = DATASET_REGISTRY['cic_ids2018']['X_test']
    if os.path.exists(cic2018_path):
        logger.info("\n── Evaluating on CIC-IDS2018 ──")
        results['cic_ids2018'] = evaluate_on_dataset(clf, 'cic_ids2018')
    else:
        logger.warning("CIC-IDS2018 splits not found — skipping")
    return results
if __name__ == "__main__":
    run()
