import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.feature_config import FEATURE_NAMES
from utils.metrics_utils import compute_ece
from utils.plot_utils import plot_confidence_distribution, plot_calibration_curve
from utils.logger import get_logger
import yaml
logger = get_logger(__name__)
MODEL_PATH    = "models/classifier/saved/decision_tree_model.pkl"
X_TEST_PATH   = "data/splits/local/X_test.csv"
Y_TEST_PATH   = "data/splits/local/y_test.csv"
METRICS_OUT   = "results/metrics/calibration_metrics.json"
SUMMARY_OUT   = "results/metrics/trustworthy_summary.json"
CONF_FIG_OUT  = "results/figures/confidence_distribution.png"
CAL_FIG_OUT   = "results/figures/calibration_curve.png"
def run(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cal_cfg = config['calibration']
    n_bins  = cal_cfg['n_bins']
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    meta_cols = ['label', 'label_str']
    available = [c for c in X_test.columns if c not in meta_cols]
    X_arr = X_test[available].values
    proba      = clf.predict_proba(X_arr)
    y_pred     = clf.predict(X_arr)
    confidence = np.max(proba, axis=1)
    correct    = (y_pred == y_test)
    routing_threshold = config['hybrid']['confidence_threshold']
    logger.info("Computing confidence distribution")
    os.makedirs(os.path.dirname(CONF_FIG_OUT), exist_ok=True)
    plot_confidence_distribution(confidence, correct,
                                  routing_threshold, CONF_FIG_OUT)
    conf_stats = {
        "mean_confidence_correct":   round(float(confidence[correct].mean()), 4),
        "mean_confidence_incorrect": round(float(confidence[~correct].mean()), 4)
        if (~correct).any() else 0.0,
        "pct_above_routing_threshold": round(
            float((confidence >= routing_threshold).mean()), 4),
        "pct_routed_to_autoencoder": round(
            float((confidence < routing_threshold).mean()), 4),
    }
    logger.info("Computing calibration curve")
    is_correct   = correct.astype(int)
    frac_correct, mean_conf = calibration_curve(
        is_correct, confidence, n_bins=n_bins, strategy='uniform')
    os.makedirs(os.path.dirname(CAL_FIG_OUT), exist_ok=True)
    plot_calibration_curve(frac_correct, mean_conf, CAL_FIG_OUT, n_bins)
    logger.info("Computing Expected Calibration Error (ECE)")
    calibration_scores = proba[:, 1] if proba.shape[1] == 2 else confidence
    ece, bin_data = compute_ece(y_test, y_pred, calibration_scores, n_bins)
    if ece < 0.05:
        interpretation = "Well calibrated — confidence reliably reflects accuracy"
    elif ece < 0.10:
        interpretation = "Moderately calibrated — minor overconfidence present"
    elif ece < 0.15:
        interpretation = (
            "Acceptable calibration — confidence routing threshold "
            "accounts for gap; future work: Platt scaling")
    else:
        interpretation = (
            "Overconfident model — confidence scores inflate routing "
            "reliability; recommend Platt scaling in future work")
    cal_metrics = {
        "ece":              round(ece, 4),
        "ece_interpretation": interpretation,
        "n_bins":           n_bins,
        "confidence_stats": conf_stats,
        "bin_details":      bin_data
    }
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(cal_metrics, f, indent=2)
    logger.info(f"Calibration metrics saved → {METRICS_OUT}")
    trustworthy = {
        "components": [
            {
                "component":    "Confidence-Based Routing",
                "tai_principle": "Uncertainty Quantification",
                "tai_category":  "Reliability",
                "measured_by":   "ECE, confidence distribution, routing rate"
            },
            {
                "component":    "Adaptive Confidence Threshold",
                "tai_principle": "Distribution Robustness",
                "tai_category":  "Robustness",
                "measured_by":   "Performance under traffic variation"
            },
            {
                "component":    "SHAP Explanations",
                "tai_principle": "Post-Hoc Interpretability",
                "tai_category":  "Transparency",
                "measured_by":   "SHAP summary plots, per-class attribution"
            },
            {
                "component":    "Temporal Aggregator",
                "tai_principle": "Prediction Stability",
                "tai_category":  "Robustness",
                "measured_by":   "Alert latency, FPR under noise"
            }
        ],
        "calibration_summary": {
            "ece":               round(ece, 4),
            "interpretation":    interpretation,
            "routing_threshold": routing_threshold,
            "pct_to_autoencoder": conf_stats['pct_routed_to_autoencoder']
        }
    }
    with open(SUMMARY_OUT, 'w') as f:
        json.dump(trustworthy, f, indent=2)
    logger.info(f"Trustworthy AI summary saved → {SUMMARY_OUT}")
    print("\n" + "=" * 60)
    print("CALIBRATION REPORT — FOR PAPER SECTION 6")
    print("=" * 60)
    print(f"ECE:                         {ece:.4f}")
    print(f"Interpretation:              {interpretation}")
    print(f"Mean conf (correct):         {conf_stats['mean_confidence_correct']:.4f}")
    print(f"Mean conf (incorrect):       {conf_stats['mean_confidence_incorrect']:.4f}")
    print(f"Flows handled by classifier: "
          f"{conf_stats['pct_above_routing_threshold']*100:.1f}%")
    print(f"Flows routed to autoencoder: "
          f"{conf_stats['pct_routed_to_autoencoder']*100:.1f}%")
    print("=" * 60)
    return cal_metrics
if __name__ == "__main__":
    run()
