import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.feature_config import FEATURE_NAMES
from utils.plot_utils import plot_mi_feature_ranking
from utils.logger import get_logger
import yaml
logger = get_logger(__name__)
INPUT_PATH   = "data/processed/local/flows_all_features.csv"
OUTPUT_PATH  = "data/processed/local/flows_mi_selected.csv"
METRICS_OUT  = "results/metrics/feature_selection_comparison.json"
FIGURE_OUT   = "results/figures/mi_feature_ranking.png"
def run(input_path=INPUT_PATH, output_path=OUTPUT_PATH,
        config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    n_features = config['feature_selection']['n_features']
    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    meta_cols    = ['label', 'label_str']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    y            = df['label'].values
    X            = df[feature_cols].values
    logger.info(f"Computing Mutual Information for {len(feature_cols)} features")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    ranking = sorted(zip(feature_cols, mi_scores),
                     key=lambda x: x[1], reverse=True)
    logger.info("\nFeature ranking by Mutual Information:")
    for rank, (feat, score) in enumerate(ranking, 1):
        marker = " ✓ SELECTED" if rank <= n_features else ""
        logger.info(f"  {rank:2d}. {feat:<30} {score:.5f}{marker}")
    selected_features = [feat for feat, _ in ranking[:n_features]]
    logger.info(f"\nSelected {n_features} features: {selected_features}")
    out_df = df[selected_features + ['label', 'label_str']].copy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info(f"MI-selected data saved → {output_path}")
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    comparison = {
        "all_features_count":      len(feature_cols),
        "selected_features_count": n_features,
        "selected_features":       selected_features,
        "full_ranking": [
            {"rank": i+1, "feature": feat, "mi_score": round(float(score), 6)}
            for i, (feat, score) in enumerate(ranking)
        ]
    }
    with open(METRICS_OUT, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Feature ranking saved → {METRICS_OUT}")
    os.makedirs(os.path.dirname(FIGURE_OUT), exist_ok=True)
    all_feat_names  = [feat for feat, _ in ranking]
    all_feat_scores = [score for _, score in ranking]
    plot_mi_feature_ranking(all_feat_names, all_feat_scores, FIGURE_OUT)
    return selected_features, comparison
if __name__ == "__main__":
    run()
