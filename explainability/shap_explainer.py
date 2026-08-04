   
                 
                                                      
                                                                   

                                       
                                                   
   

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.feature_config import FEATURE_NAMES, CLASS_NAMES_LOCAL
from utils.shap_utils import (compute_shap_values,
                               mean_abs_shap_per_class,
                               top_features_per_class)
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)

MODEL_PATH   = "models/classifier/saved/decision_tree_model.pkl"
X_TRAIN_PATH = "data/splits/local/X_train.csv"
X_TEST_PATH  = "data/splits/local/X_test.csv"
SHAP_OUT     = "results/metrics/shap_values.npy"
ATTR_OUT     = "results/metrics/shap_class_attribution.json"


def run(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    n_background = config['explainability']['background_samples']

    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)
    meta_cols = ['label', 'label_str']
    available = [c for c in X_train.columns if c not in meta_cols]

    X_bg   = X_train[available].values[:n_background]
    X_exp  = X_test[available].values

    logger.info(f"Background samples: {len(X_bg)} | "
                f"Explain samples: {len(X_exp)}")

    try:
        explainer, shap_values = compute_shap_values(clf, X_bg, X_exp)
    except ImportError:
        logger.error("shap not installed — run: pip install shap")
        return None

                          
    os.makedirs(os.path.dirname(SHAP_OUT), exist_ok=True)
    np.save(SHAP_OUT, shap_values)
    logger.info(f"SHAP values saved → {SHAP_OUT}")

                           
    class_attr = mean_abs_shap_per_class(shap_values, CLASS_NAMES_LOCAL, available)
    top_feats  = top_features_per_class(class_attr, top_n=3)

    output = {
        "class_attribution": class_attr,
        "top_3_features_per_class": {
            cls: [{"feature": f, "mean_abs_shap": v} for f, v in feats]
            for cls, feats in top_feats.items()
        }
    }

    with open(ATTR_OUT, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"SHAP class attribution saved → {ATTR_OUT}")

                                                       
    print("\nTop 3 Features per Attack Class (SHAP):")
    for cls, feats in top_feats.items():
        feat_str = ", ".join([f"{f} ({v:.4f})" for f, v in feats])
        print(f"  {cls}: {feat_str}")

    return shap_values, class_attr


if __name__ == "__main__":
    run()
