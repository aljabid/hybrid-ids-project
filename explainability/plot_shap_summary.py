import os
import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from utils.feature_config import FEATURE_NAMES, CLASS_NAMES_LOCAL
from utils.logger import get_logger
import yaml
logger = get_logger(__name__)
SHAP_PATH    = "results/metrics/shap_values.npy"
ATTR_PATH    = "results/metrics/shap_class_attribution.json"
X_TEST_PATH  = "data/splits/local/X_test.csv"
SUMMARY_OUT  = "results/figures/shap_summary_plot.png"
PERCLASS_OUT = "results/figures/shap_per_class_plot.png"
def plot_per_class_bar(class_attr, features, output_path):
    class_names = sorted(CLASS_NAMES_LOCAL)
    n_classes  = len(class_names)
    n_features = len(features)
    x = np.arange(n_features)
    width = 0.2
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cls in enumerate(class_names):
        if cls not in class_attr:
            continue
        vals = [
            class_attr[cls].get(f, 0)
            for f in features
        ]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=cls,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="white"
        )
    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels(
        features,
        rotation=35,
        ha="right",
        fontsize=9
    )
    ax.set_ylabel("Mean |SHAP Value|", fontweight="bold")
    ax.set_title("Feature Importance by Class (SHAP)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP per-class plot saved → {output_path}")
def run(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    max_display = config["explainability"]["max_display"]
    if not os.path.exists(SHAP_PATH):
        logger.error("SHAP values not found — run shap_explainer first")
        return
    shap_values = np.load(SHAP_PATH, allow_pickle=True)
    X_test = pd.read_csv(X_TEST_PATH)
    available_features = [
        f for f in FEATURE_NAMES
        if f in X_test.columns
    ]
    X_arr = X_test[available_features].values
    with open(ATTR_PATH) as f:
        attr_data = json.load(f)
    class_attr = attr_data["class_attribution"]
    try:
        import shap
        os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
        if isinstance(shap_values, list):
            shap_array = np.mean(np.array(shap_values), axis=0)
        else:
            shap_array = np.array(shap_values)
        logger.info(f"SHAP array shape: {shap_array.shape}")
        plt.figure()
        shap.summary_plot(
            shap_array,
            X_arr,
            feature_names=available_features,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            SUMMARY_OUT,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        logger.info(f"SHAP summary saved → {SUMMARY_OUT}")
    except Exception as e:
        logger.warning(f"SHAP summary failed: {e}")
    os.makedirs(os.path.dirname(PERCLASS_OUT), exist_ok=True)
    plot_per_class_bar(
        class_attr,
        available_features,
        PERCLASS_OUT
    )
if __name__ == "__main__":
    run()
