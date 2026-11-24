import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.insert(0, ROOT_DIR)
from utils.metrics_utils import full_metrics_dict
from utils.plot_utils import plot_confusion_matrix
from utils.dataset_utils import DATASET_REGISTRY
from utils.logger import get_logger
logger = get_logger(__name__)
MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "classifier",
    "saved",
    "decision_tree_model.pkl"
)
def create_local_splits_if_missing():
    x_test_path = os.path.join(
        ROOT_DIR,
        "data",
        "splits",
        "local",
        "X_test.csv"
    )
    y_test_path = os.path.join(
        ROOT_DIR,
        "data",
        "splits",
        "local",
        "y_test.csv"
    )
    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        logger.info("Local splits already exist.")
        return
    logger.warning(
        "Local train/test splits missing. "
        "Attempting automatic creation..."
    )
    raw_candidates = [
        os.path.join(
            ROOT_DIR,
            "data",
            "processed",
            "local",
            "normalized.csv"
        ),
        os.path.join(
            ROOT_DIR,
            "data",
            "local.csv"
        )
    ]
    dataset_path = None
    for p in raw_candidates:
        if os.path.exists(p):
            dataset_path = p
            break
    if dataset_path is None:
        logger.error(
            "No local dataset found for split creation."
        )
        return
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        logger.error(
            "'label' column missing from dataset."
        )
        return
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    split_dir = os.path.join(
        ROOT_DIR,
        "data",
        "splits",
        "local"
    )
    os.makedirs(split_dir, exist_ok=True)
    X_train.to_csv(
        os.path.join(split_dir, "X_train.csv"),
        index=False
    )
    X_test.to_csv(
        os.path.join(split_dir, "X_test.csv"),
        index=False
    )
    y_train.to_csv(
        os.path.join(split_dir, "y_train.csv"),
        index=False
    )
    y_test.to_csv(
        os.path.join(split_dir, "y_test.csv"),
        index=False
    )
    logger.info(
        "Local train/test splits created successfully."
    )
def train_model_if_missing():
    if os.path.exists(MODEL_PATH):
        logger.info("Classifier model found.")
        return
    logger.warning(
        "Classifier model missing. "
        "Training automatically..."
    )
    train_x = os.path.join(
        ROOT_DIR,
        "data",
        "splits",
        "local",
        "X_train.csv"
    )
    train_y = os.path.join(
        ROOT_DIR,
        "data",
        "splits",
        "local",
        "y_train.csv"
    )
    if not (
        os.path.exists(train_x)
        and os.path.exists(train_y)
    ):
        logger.error(
            "Training data missing."
        )
        return
    X_train = pd.read_csv(train_x)
    y_train = pd.read_csv(train_y).values.ravel()
    clf = DecisionTreeClassifier(
        random_state=42
    )
    clf.fit(X_train, y_train)
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    logger.info(
        f"Model saved → {MODEL_PATH}"
    )
def evaluate_on_dataset(clf, dataset_name):
    entry = DATASET_REGISTRY[dataset_name]
    if not (
        os.path.exists(entry["X_test"])
        and os.path.exists(entry["y_test"])
    ):
        logger.warning(
            f"{dataset_name} test split missing."
        )
        return None
    X = pd.read_csv(entry["X_test"])
    y = pd.read_csv(
        entry["y_test"]
    ).values.ravel()
    meta_cols = ["label", "label_str"]
    available = [
        c for c in X.columns
        if c not in meta_cols
    ]
    X = X[available]
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)
    metrics = full_metrics_dict(
        y,
        y_pred,
        dataset_name=dataset_name
    )
    try:
        if entry["binary"]:
            auc_score = roc_auc_score(
                y,
                proba[:, 1]
            )
        else:
            auc_score = roc_auc_score(
                y,
                proba,
                multi_class="ovr",
                average="weighted"
            )
        metrics["auc"] = round(
            auc_score,
            4
        )
    except Exception as e:
        logger.warning(
            f"AUC error: {e}"
        )
        metrics["auc"] = 0.0
    metrics["classification_report"] = (
        classification_report(
            y,
            y_pred,
            zero_division=0
        )
    )
    out_path = entry[
        "metrics_out"
    ].replace(
        "hybrid_",
        "classifier_"
    )
    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )
    with open(out_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2
        )
    plot_confusion_matrix(
        y,
        y_pred,
        np.unique(y),
        title=f"Confusion Matrix — {dataset_name}",
        output_path=(
            f"results/figures/"
            f"confusion_matrix_classifier_"
            f"{dataset_name}.png"
        )
    )
    logger.info(
        f"{dataset_name} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"AUC={metrics['auc']:.4f}"
    )
    return metrics
def run():
    logger.info(
        "Preparing evaluation..."
    )
    create_local_splits_if_missing()
    train_model_if_missing()
    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Evaluation aborted. "
            "Model unavailable."
        )
        return None
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    results = {}
    for ds in [
        "local",
        "cicids2017",
        "cic_ids2018"
    ]:
        logger.info(
            f"\n── Evaluating {ds} ──"
        )
        result = evaluate_on_dataset(
            clf,
            ds
        )
        if result:
            results[ds] = result
    return results
if __name__ == "__main__":
    run()
