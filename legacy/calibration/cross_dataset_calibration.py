import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
from calibration.calibration_analysis import (
    compute_ece, get_binary_proba, load_dataset
)
MODEL_PATH  = "models/classifier/saved/decision_tree_model.pkl"
TRAIN_X     = "data/splits/train/X_train.csv"
TRAIN_Y     = "data/splits/train/y_train.csv"
CICIDS_X    = "data/splits/cicids2017/X_test_cicids.csv"
CICIDS_Y    = "data/splits/cicids2017/y_test_cicids.csv"
IDS2018_X   = "data/splits/cic_ids2018/X_test_cic2018.csv"
IDS2018_Y   = "data/splits/cic_ids2018/y_test_cic2018.csv"
FIG_OUT     = "results/figures/figure5_cross_dataset_calibration.png"
METRICS_OUT = "results/metrics/cross_dataset_calibration.json"
def run():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found — run main.py first")
        return None
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    datasets = [
        ("CICIDS2017 (In-Domain)",    CICIDS_X,  CICIDS_Y,  '#2196F3', 'o-'),
        ("CIC-IDS2018 (Cross-Domain)",IDS2018_X, IDS2018_Y, '#FF9800', 's-'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        'Cross-Dataset Calibration Comparison\n'
        'Does the classifier remain calibrated under domain shift?',
        fontsize=13, fontweight='bold'
    )
    report = {}
    for name, x_path, y_path, color, marker in datasets:
        X, y   = load_dataset(x_path, y_path, name)
        y_prob = get_binary_proba(clf, X)
        ece, bin_data = compute_ece(y, y_prob, n_bins=10)
        frac_pos, mean_pred = calibration_curve(
            y, y_prob, n_bins=10, strategy='uniform')
        fpr_arr, tpr_arr, _ = roc_curve(y, y_prob)
        auc_score = float(auc(fpr_arr, tpr_arr))
        report[name] = {
            "ece":       round(ece, 4),
            "auc":       round(auc_score, 4),
            "n_samples": int(len(y)),
        }
        axes[0].plot(mean_pred, frac_pos, marker, color=color,
                     linewidth=2.5, markersize=7,
                     label=f'{name}\nECE={ece:.4f}')
        axes[1].plot(fpr_arr, tpr_arr, color=color, linewidth=2,
                     label=f'{name}\nAUC={auc_score:.3f}')
        axes[2].hist(y_prob, bins=40, density=True, alpha=0.55,
                     color=color, label=name, edgecolor='none')
    axes[0].plot([0,1],[0,1], 'k--', alpha=0.5, linewidth=1.5,
                 label='Perfect calibration')
    axes[0].set_xlabel('Mean Predicted Probability', fontweight='bold')
    axes[0].set_ylabel('Fraction of Positives',      fontweight='bold')
    axes[0].set_title('A — Reliability Diagrams\n(closer to diagonal = better)',
                      fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1])
    axes[0].grid(alpha=0.3)
    axes[1].plot([0,1],[0,1], 'k--', alpha=0.4, linewidth=1)
    axes[1].set_xlabel('False Positive Rate', fontweight='bold')
    axes[1].set_ylabel('True Positive Rate',  fontweight='bold')
    axes[1].set_title('B — ROC Curves',       fontweight='bold')
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(alpha=0.3)
    axes[2].axvline(0.5, color='black', linestyle='--',
                    linewidth=1.5, alpha=0.7, label='Decision boundary')
    axes[2].set_xlabel('Predicted P(Attack)',   fontweight='bold')
    axes[2].set_ylabel('Density',               fontweight='bold')
    axes[2].set_title('C — Confidence Distributions\n(separation = discrimination)',
                      fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Cross-dataset calibration figure saved → {FIG_OUT}")
    ece_vals = [v['ece'] for v in report.values()]
    if len(ece_vals) == 2:
        ece_degradation = ece_vals[1] - ece_vals[0]
        report['ece_degradation'] = round(ece_degradation, 4)
        report['ece_degradation_pct'] = round(
            ece_degradation / max(ece_vals[0], 1e-6) * 100, 1)
        logger.info(f"\nECE degradation: "
                    f"{ece_vals[0]:.4f} → {ece_vals[1]:.4f} "
                    f"(+{ece_degradation:.4f})")
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Cross-dataset calibration saved → {METRICS_OUT}")
    return report
if __name__ == "__main__":
    run()
