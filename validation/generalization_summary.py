   
                         
                                               
                                                                

                                                     
                                                   
   

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.plot_utils import plot_generalization_comparison
from utils.logger import get_logger

logger = get_logger(__name__)

DATASET_LABELS = ["Local (Train Domain)", "CICIDS2017", "CIC-IDS2018"]
METRIC_FILES   = [
    "results/metrics/hybrid_metrics_local.json",
    "results/metrics/hybrid_metrics_cicids.json",
    "results/metrics/hybrid_metrics_cic2018.json",
]
METRICS_PLOT  = ["accuracy", "f1", "tpr", "fpr"]
FIGURE_OUT    = "results/figures/generalization_comparison.png"
SUMMARY_OUT   = "results/metrics/generalization_summary.json"


def run():
    results       = []
    valid_labels  = []

    for label, path in zip(DATASET_LABELS, METRIC_FILES):
        if not os.path.exists(path):
            logger.warning(f"Not found: {path} — skipping {label}")
            continue
        with open(path) as f:
            results.append(json.load(f))
        valid_labels.append(label)

    if not results:
        logger.error("No hybrid metric files found — run evaluate_hybrid.py first")
        return None

          
    os.makedirs(os.path.dirname(FIGURE_OUT), exist_ok=True)
    plot_generalization_comparison(valid_labels, results, METRICS_PLOT, FIGURE_OUT)

                            
    local_acc = float(results[0].get('accuracy', 0))
    drops     = {}
    for label, r in zip(valid_labels[1:], results[1:]):
        drop = local_acc - float(r.get('accuracy', 0))
        drops[label] = round(drop, 4)
        logger.info(f"Accuracy drop (Local → {label}): {drop:.4f} "
                    f"({drop/local_acc*100:.1f}% relative)")

                           
    print("\n" + "=" * 75)
    print("GENERALIZATION TABLE — PAPER RESULTS SECTION")
    print("=" * 75)
    print(f"{'Dataset':<25} {'Accuracy':>9} {'F1':>8} {'TPR':>8} {'FPR':>8} {'AUC':>8}")
    print("-" * 75)
    for label, r in zip(valid_labels, results):
        print(f"{label:<25} "
              f"{r.get('accuracy',0):>9.4f} "
              f"{r.get('f1',0):>8.4f} "
              f"{r.get('tpr',0):>8.4f} "
              f"{r.get('fpr',0):>8.4f} "
              f"{r.get('auc',0):>8.4f}")
    print("=" * 75)

    if drops:
        print("\nAccuracy drop from local domain:")
        for label, drop in drops.items():
            print(f"  Local → {label}: -{drop:.4f}")

    summary = {
        "datasets":                valid_labels,
        "results":                 results,
        "accuracy_drop_from_local": drops
    }

    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    with open(SUMMARY_OUT, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Generalization summary saved → {SUMMARY_OUT}")

    return summary


if __name__ == "__main__":
    run()
