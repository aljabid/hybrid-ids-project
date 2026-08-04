   
                   
                                                           
                                                          
                               

                                     
                                     
                                                            
                                                       
   

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.feature_config import FEATURE_NAMES, CLASS_NAMES_LOCAL
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)

X_TRAIN_PATH  = "data/splits/local/X_train.csv"
Y_TRAIN_PATH  = "data/splits/local/y_train.csv"
MODEL_OUT     = "models/classifier/saved/decision_tree_model.pkl"
PARAMS_PATH   = "models/classifier/saved/best_params.json"


def train(config_path="config/config.yaml"):

    with open(config_path) as f:
        config = yaml.safe_load(f)

                                                             
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        logger.info(f"Loaded tuned params: {params}")
    else:
        params = {
            "criterion":         config['classifier']['criterion'],
            "max_depth":         config['classifier']['max_depth'],
            "min_samples_split": config['classifier']['min_samples_split'],
        }
        logger.info(f"Using config default params: {params}")

    random_state = config['classifier']['random_state']

    logger.info("Loading training data")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()

    meta_cols = ['label', 'label_str']
    available = [c for c in X_train.columns if c not in meta_cols]
    X_train = X_train[available].values

    logger.info(f"Training samples: {len(X_train)} | Features: {len(available)}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

                                                               
    clf = DecisionTreeClassifier(
        criterion=params.get('criterion', 'gini'),
        max_depth=params.get('max_depth', 10),
        min_samples_split=params.get('min_samples_split', 5),
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    logger.info(f"Training complete | Tree depth: {clf.get_depth()} | "
                f"Leaves: {clf.get_n_leaves()}")

                                                                
    proba_sample = clf.predict_proba(X_train[:5])
    logger.info(f"predict_proba sample shape: {proba_sample.shape} ✓")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved → {MODEL_OUT}")

    return clf


if __name__ == "__main__":
    train()
