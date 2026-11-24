import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hybrid.hybrid_detector import run_hybrid
from utils.metrics_utils import full_metrics_dict, compute_roc_auc
from utils.plot_utils import plot_confusion_matrix
from utils.dataset_utils import DATASET_REGISTRY
from utils.logger import get_logger
logger = get_logger(__name__)
def evaluate_dataset(dataset_name):
    entry = DATASET_REGISTRY[dataset_name]
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating hybrid system on: {dataset_name}")
    logger.info(f"{'='*50}")
    predictions, y_true, log = run_hybrid(dataset_name=dataset_name)
    y_binary = (y_true != 0).astype(int)
    predictions = (predictions != 0).astype(int)
    metrics = full_metrics_dict(y_binary, predictions,
                                dataset_name=dataset_name)
    try:
        conf_scores = np.array([m.get('confidence', 0.5) for m in log])
        try:
            metrics['auc'] = round(compute_roc_auc(y_binary, 1 - conf_scores), 4)
        except Exception:
            metrics['auc'] = 0.0
    except Exception:
        metrics['auc'] = 0.0
    metrics['routing_pct_classifier'] = round(
        float(np.mean([m['module'] == 'classifier' for m in log])), 4)
    metrics['routing_pct_autoencoder'] = round(
        float(np.mean([m['module'] == 'autoencoder' for m in log])), 4)
    out_path = entry['metrics_out']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {out_path}")
    plot_confusion_matrix(
        y_binary, predictions, entry['labels'],
        title=f"Confusion Matrix (Hybrid) — {dataset_name}",
        output_path=entry['cm_out']
    )
    logger.info(f"Results | Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"TPR={metrics['tpr']:.4f} | "
                f"FPR={metrics['fpr']:.4f}")
    return metrics
def run():
    results = {}
    for ds in ['local', 'cicids2017', 'cic_ids2018']:
        x_path = DATASET_REGISTRY[ds]['X_test']
        if os.path.exists(x_path):
            results[ds] = evaluate_dataset(ds)
        else:
            logger.warning(f"{ds} test splits not found — skipping")
    print("\n" + "=" * 65)
    print("HYBRID SYSTEM — RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Dataset':<20} {'Acc':>8} {'F1':>8} {'TPR':>8} {'FPR':>8}")
    print("-" * 65)
    for ds, m in results.items():
        print(f"{ds:<20} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
              f"{m['tpr']:>8.4f} {m['fpr']:>8.4f}")
    print("=" * 65)
    return results
if __name__ == "__main__":
    run()
