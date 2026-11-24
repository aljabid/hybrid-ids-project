import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, confusion_matrix, classification_report
)
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
CICIDS_X    = "data/splits/cicids2017/X_test_cicids.csv"
CICIDS_Y    = "data/splits/cicids2017/y_test_cicids.csv"
IDS2018_X   = "data/splits/cic_ids2018/X_test_cic2018.csv"
IDS2018_Y   = "data/splits/cic_ids2018/y_test_cic2018.csv"
FIG_OUT     = "results/figures/figure6_cross_dataset_results.png"
TABLE_CSV   = "results/metrics/cross_dataset_results_table.csv"
TABLE_JSON  = "results/metrics/cross_dataset_results_table.json"
def full_metrics(y_true, y_pred, y_prob, name):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        tpr = float(recall_score(y_true, y_pred, zero_division=0))
        fpr = 0.0
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    auc_score = float(auc(fpr_arr, tpr_arr))
    ece, _ = compute_ece(y_true, y_prob, n_bins=10)
    return {
        "dataset":    name,
        "n_samples":  int(len(y_true)),
        "accuracy":   round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":         round(float(f1_score(y_true, y_pred,
                                            zero_division=0)), 4),
        "precision":  round(float(precision_score(y_true, y_pred,
                                                   zero_division=0)), 4),
        "tpr":        round(float(tpr), 4),
        "fpr":        round(float(fpr), 4),
        "auc":        round(auc_score, 4),
        "ece":        round(ece, 4),
        "fpr_curve":  fpr_arr,
        "tpr_curve":  tpr_arr,
    }
def run():
    if not os.path.exists(MODEL_PATH):
        logger.error("Model not found — run main.py first")
        return None
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    logger.info(f"Model loaded: depth={clf.get_depth()}")
    datasets = [
        ("CICIDS2017 (In-Domain)",     CICIDS_X,  CICIDS_Y,  '#2196F3'),
        ("CIC-IDS2018 (Cross-Domain)", IDS2018_X, IDS2018_Y, '#FF9800'),
    ]
    results = []
    for name, x_path, y_path, color in datasets:
        X, y   = load_dataset(x_path, y_path, name)
        y_prob = get_binary_proba(clf, X)
        y_pred = (clf.predict(X) != 0).astype(int)
        m = full_metrics(y, y_pred, y_prob, name)
        m['color'] = color
        results.append(m)
        logger.info(f"\n{name}:")
        logger.info(f"  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
                    f"TPR={m['tpr']:.4f}  FPR={m['fpr']:.4f}  "
                    f"AUC={m['auc']:.4f}  ECE={m['ece']:.4f}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Cross-Dataset Evaluation Results',
                 fontsize=14, fontweight='bold')
    ax = axes[0]
    for m in results:
        ax.plot(m['fpr_curve'], m['tpr_curve'],
                color=m['color'], linewidth=2.5,
                label=f"{m['dataset'].split('(')[0].strip()} "
                      f"(AUC={m['auc']:.3f})")
    ax.plot([0,1],[0,1], 'k--', alpha=0.4, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontweight='bold')
    ax.set_title('A — ROC Curves',       fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax = axes[1]
    metrics_plot = ['accuracy', 'f1', 'tpr', 'auc']
    labels_plot  = ['Accuracy', 'F1', 'TPR', 'AUC']
    x = np.arange(len(metrics_plot))
    w = 0.35
    for i, m in enumerate(results):
        vals = [m[k] for k in metrics_plot]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w,
                      color=m['color'], alpha=0.85,
                      label=m['dataset'].split('(')[0].strip(),
                      edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('B — Performance Metrics Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax = axes[2]
    metrics_c = ['fpr', 'ece']
    labels_c  = ['FPR', 'ECE']
    x = np.arange(len(metrics_c))
    for i, m in enumerate(results):
        vals = [m[k] for k in metrics_c]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w,
                      color=m['color'], alpha=0.85,
                      label=m['dataset'].split('(')[0].strip(),
                      edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_c, fontweight='bold')
    ax.set_ylabel('Rate', fontweight='bold')
    ax.set_title('C — Error and Calibration\n(lower is better)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Cross-dataset figure saved → {FIG_OUT}")
    table_cols = ['dataset', 'n_samples', 'accuracy', 'f1',
                  'precision', 'tpr', 'fpr', 'auc', 'ece']
    rows = [{c: m[c] for c in table_cols} for m in results]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_CSV, index=False)
    logger.info(f"Results table saved → {TABLE_CSV}")
    print("\n" + "="*80)
    print("CROSS-DATASET RESULTS TABLE — FOR PAPER")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    with open(TABLE_JSON, 'w') as f:
        json.dump(rows, f, indent=2)
    return rows
if __name__ == "__main__":
    run()
