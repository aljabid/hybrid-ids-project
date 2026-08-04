   
                               

                                                                          
                                                        

                                                                    
                                                     

         
                                        
                                         
                                                    
                                                     
   

import os
import sys
import json
import pickle

                                                              
                                                              
                                                               
                                                
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')

MODEL_PATH = "models/classifier/saved/decision_tree_model.pkl"
TRAIN_X = "data/splits/train/X_train.csv"
CICIDS_X = "data/splits/cicids2017/X_test_cicids.csv"
CICIDS_Y = "data/splits/cicids2017/y_test_cicids.csv"
IDS2018_X = "data/splits/cic_ids2018/X_test_cic2018.csv"
IDS2018_Y = "data/splits/cic_ids2018/y_test_cic2018.csv"

SHAP_CICIDS_OUT = "results/metrics/shap_values_cicids.npy"
SHAP_IDS18_OUT = "results/metrics/shap_values_ids2018.npy"
ATTR_CICIDS_OUT = "results/metrics/shap_class_attribution_cicids.json"
ATTR_IDS18_OUT = "results/metrics/shap_class_attribution_ids2018.json"


def _load_or_synthetic(x_path, y_path, name, n_features=10):
                                                                           
    if os.path.exists(x_path):
        X = pd.read_csv(x_path)
        meta = ['label', 'label_str']
        feature_cols = [c for c in X.columns if c not in meta]
        y = pd.read_csv(y_path).values.ravel() if os.path.exists(y_path) else None
        logger.info(f"Loaded {name}: {X.shape[0]} rows")
        return X[feature_cols].values, y, feature_cols
    else:
        logger.warning(f"{name} not found — using synthetic demo data")
        np.random.seed(hash(name) % 1000)
        X = np.random.rand(500, n_features)
        y = np.random.randint(0, 2, 500)
        features = [f'feature_{i}' for i in range(n_features)]
        return X, y, features


def compute_shap_values(clf, X_background, X_explain, n_background=100):
       
                                             
                                                                         

               
                                                   
                                                                     
                                      
                                                          

            
                                                                            
                                                       
       
    try:
        import shap
    except ImportError:
        logger.error("shap not installed — run: pip install shap")
        raise

    bg = X_background[:n_background]
    logger.info(f"Computing SHAP — background: {len(bg)} | "
                f"explain: {len(X_explain)}")

    explainer = shap.TreeExplainer(clf, bg)
    shap_values = explainer.shap_values(X_explain)

    logger.info("SHAP computation complete")
    return shap_values


def mean_abs_shap(shap_values, feature_names):
       
                                                  
                                                                      

            
                                                 
       
                                                                  
    if isinstance(shap_values, list):
                                             
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_vals = stacked.mean(axis=0).mean(axis=0)
    else:
                                                          
        sv_arr = np.array(shap_values)
        if sv_arr.ndim == 3:
                                                                     
            sv_arr = sv_arr.mean(axis=0)
        mean_vals = np.abs(sv_arr).mean(axis=0)

    return {feat: round(float(v), 6)
            for feat, v in zip(feature_names, mean_vals)}


def run():
                     
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Run main.py first to train the classifier")
        return None

    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    logger.info(f"Classifier loaded: depth={clf.get_depth()}")

                                            
    X_train, _, feature_names = _load_or_synthetic(
        TRAIN_X, None, 'training')

                        
    X_cicids, y_cicids, feat_c = _load_or_synthetic(
        CICIDS_X, CICIDS_Y, 'CICIDS2017')
    X_ids18, y_ids18, feat_i = _load_or_synthetic(
        IDS2018_X, IDS2018_Y, 'CIC-IDS2018')

                                  
    features = feat_c if feat_c else feature_names

    os.makedirs(os.path.dirname(SHAP_CICIDS_OUT), exist_ok=True)

    results = {}

    for name, X, y, shap_out, attr_out in [
        ('CICIDS2017', X_cicids, y_cicids, SHAP_CICIDS_OUT, ATTR_CICIDS_OUT),
        ('CIC-IDS2018', X_ids18, y_ids18, SHAP_IDS18_OUT, ATTR_IDS18_OUT),
    ]:
        logger.info(f"\n── SHAP for {name} ──")
        try:
            shap_vals = compute_shap_values(clf, X_train, X)
        except ImportError:
            logger.warning("shap library not installed — skipping SHAP computation")
            logger.warning("Install with: pip install shap")
            return None

                              
        np.save(shap_out, shap_vals)
        logger.info(f"SHAP values saved → {shap_out}")

                             
        attribution = mean_abs_shap(shap_vals, features)

                            
        sorted_attr = dict(sorted(attribution.items(),
                                  key=lambda x: x[1], reverse=True))

        with open(attr_out, 'w') as f:
            json.dump(sorted_attr, f, indent=2)
        logger.info(f"Attribution saved → {attr_out}")

        logger.info(f"Top 5 features ({name}):")
        for feat, val in list(sorted_attr.items())[:5]:
            logger.info(f"  {feat:<28} {val:.6f}")

        results[name] = sorted_attr

    return results


if __name__ == "__main__":
    run()
