import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.feature_config import FEATURE_NAMES
from utils.logger import get_logger
import yaml
logger = get_logger(__name__)
X_TRAIN_PATH = "data/splits/local/X_train.csv"
Y_TRAIN_PATH = "data/splits/local/y_train.csv"
PARAMS_OUT   = "models/classifier/saved/best_params.json"
PARAM_GRID = {
    'max_depth':         [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
}
def tune(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    random_state = config['classifier']['random_state']
    logger.info("Loading training data for hyperparameter tuning")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    meta_cols = ['label', 'label_str']
    available = [c for c in X_train.columns if c not in meta_cols]
    X_train = X_train[available].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    clf = DecisionTreeClassifier(random_state=random_state)
    logger.info(f"Running GridSearchCV with param grid: {PARAM_GRID}")
    search = GridSearchCV(
        clf, PARAM_GRID,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_params_
    best_score = search.best_score_
    logger.info(f"Best params: {best}")
    logger.info(f"Best CV F1 (weighted): {best_score:.4f}")
    os.makedirs(os.path.dirname(PARAMS_OUT), exist_ok=True)
    with open(PARAMS_OUT, 'w') as f:
        json.dump({**best, "cv_f1_weighted": round(best_score, 4)}, f, indent=2)
    logger.info(f"Best params saved → {PARAMS_OUT}")
    return best, best_score
if __name__ == "__main__":
    tune()
