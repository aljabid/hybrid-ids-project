import os
import sys
import json
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger
logger = get_logger(__name__)
METRICS_FILES = {
    "Decision Tree Only":        "results/metrics/baseline_classifier_only.json",
    "Autoencoder Only":          "results/metrics/baseline_autoencoder_only.json",
    "Fixed-Threshold Hybrid":    "results/metrics/baseline_fixed_threshold_hybrid.json",
    "Proposed Hybrid (Ours)":    "results/metrics/hybrid_metrics_local.json",
}
TABLE_OUT   = "results/metrics/baseline_comparison_table.csv"
METRICS_KEY = ["accuracy", "precision", "recall", "f1", "tpr", "fpr", "auc"]
def run():
    rows = []
    for model_name, path in METRICS_FILES.items():
        if not os.path.exists(path):
            logger.warning(f"Metrics file not found: {path} — skipping")
            continue
        with open(path) as f:
            m = json.load(f)
        row = {"Model": model_name}
        for key in METRICS_KEY:
            row[key.upper()] = round(float(m.get(key, 0.0)), 4)
        rows.append(row)
    if not rows:
        logger.error("No metrics files found — run all baselines first")
        return None
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c != "Model"]
    os.makedirs(os.path.dirname(TABLE_OUT), exist_ok=True)
    df.to_csv(TABLE_OUT, index=False)
    logger.info(f"Comparison table saved → {TABLE_OUT}")
    print("\n" + "=" * 90)
    print("TABLE 3 — BASELINE COMPARISON (Local Dataset)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print("\nBest values per metric:")
    higher_better = ["ACCURACY", "PRECISION", "RECALL", "F1", "TPR", "AUC"]
    lower_better  = ["FPR"]
    for col in numeric_cols:
        if col in higher_better:
            best_idx = df[col].idxmax()
        else:
            best_idx = df[col].idxmin()
        print(f"  {col}: {df.loc[best_idx, 'Model']} "
              f"({df.loc[best_idx, col]:.4f})")
    return df
if __name__ == "__main__":
    run()
