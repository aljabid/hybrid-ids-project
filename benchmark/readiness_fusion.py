                                                                   

from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, matthews_corrcoef, average_precision_score,
)
from sklearn.model_selection import train_test_split

from benchmark.common import resolve, write_json
from benchmark.metrics import binary_metrics
from benchmark.readiness_anomaly import (
    empirical_probability, make_scaler, train_anomaly,
)
from benchmark.readiness_supervised import load_data


def _config() -> dict:
    with resolve("benchmark/readiness_config.yaml").open() as handle:
        return yaml.safe_load(handle)


def _attack_probability(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def _fit_calibrator(model, X: np.ndarray, y: np.ndarray, method: str):
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X, y)
    return calibrated


def _entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def select(seed: int = 42) -> dict:
    config = _config()
    paths = {key: resolve(value) for key, value in config["paths"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    X_train, y_train, _, _ = load_data("train")
    X_validation, y_validation, _, _ = load_data("validation")
    rng = np.random.default_rng(seed)
    validation_cap = int(config["selection"].get("validation_selection_cap", 100000))
    per_class = validation_cap // 2
    validation_indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(y_validation == label),
                min(per_class, int((y_validation == label).sum())),
                replace=False,
            )
            for label in (0, 1)
        ]
    )
    rng.shuffle(validation_indices)
    X_validation = X_validation[validation_indices]
    y_validation = y_validation[validation_indices]
    indices = np.arange(len(y_validation))
    calibration_indices, selection_indices = train_test_split(
        indices,
        test_size=0.5,
        stratify=y_validation,
        random_state=seed,
    )
    with (
        paths["models"] / f"random_forest_seed_{seed}.pkl"
    ).open("rb") as handle:
        random_forest = pickle.load(handle)

    calibrators = {"uncalibrated": random_forest}
    for method in ("sigmoid", "isotonic"):
        calibrators[method] = _fit_calibrator(
            random_forest,
            X_validation[calibration_indices],
            y_validation[calibration_indices],
            method,
        )
    calibration_rows = []
    for name, model in calibrators.items():
        probability = _attack_probability(model, X_validation[selection_indices])
        metrics = binary_metrics(
            y_validation[selection_indices],
            probability >= 0.5,
            probability,
        )
        calibration_rows.append(
            {
                "method": name,
                "selection_brier": metrics["brier"],
                "selection_ece": metrics["ece"],
                "selection_nll": metrics["negative_log_likelihood"],
                "selection_mcc": metrics["mcc"],
                "selection_pr_auc": metrics["pr_auc"],
            }
        )
    calibration_table = pd.DataFrame(calibration_rows).sort_values(
        ["selection_brier", "selection_nll"]
    )
    calibration_table.to_csv(
        paths["tables"] / "calibration_method_selection.csv", index=False
    )
    calibration_name = str(calibration_table.iloc[0]["method"])
    calibrated_model = calibrators[calibration_name]

    anomaly_selection = json.loads(
        (paths["manifests"] / "anomaly_scaler_selection.json").read_text()
    )["winner"]
    scaler_name = anomaly_selection["scaler"]
    anomaly_name = anomaly_selection["anomaly_model"]
    scaler = make_scaler(scaler_name, seed)
    normal_train = X_train[y_train == 0]
    cap = int(config["selection"]["anomaly_training_normal_cap"])
    normal_fit = normal_train[
        rng.choice(len(normal_train), min(cap, len(normal_train)), replace=False)
    ]
    scaler.fit(normal_fit)
    detector = train_anomaly(
        anomaly_name, scaler.transform(normal_fit).astype(np.float32), seed
    )
    validation_scaled = scaler.transform(X_validation).astype(np.float32)
    reference_scores = detector.score(
        validation_scaled[calibration_indices][y_validation[calibration_indices] == 0]
    )
    anomaly_probability = empirical_probability(
        detector.score(validation_scaled), reference_scores
    )
    classifier_probability = _attack_probability(calibrated_model, X_validation)
    y_select = y_validation[selection_indices]
    classifier_select = classifier_probability[selection_indices]
    anomaly_select = anomaly_probability[selection_indices]

    candidates = []
    thresholds = np.linspace(0.05, 0.95, 37)
                                     
    for threshold in thresholds:
        prediction = classifier_select >= threshold
        candidates.append(
            {
                "family": "classifier_only",
                "parameters": {"threshold": float(threshold)},
                "mcc": float(matthews_corrcoef(y_select, prediction)),
                "pr_auc": float(average_precision_score(y_select, classifier_select)),
                "fpr": binary_metrics(y_select, prediction, classifier_select)["fpr"],
            }
        )
                                 
    for alpha in config["selection"]["fusion_weights"]:
        score = alpha * classifier_select + (1 - alpha) * anomaly_select
        for threshold in thresholds:
            prediction = score >= threshold
            metrics = binary_metrics(y_select, prediction, score)
            candidates.append(
                {
                    "family": "weighted_fusion",
                    "parameters": {
                        "alpha": float(alpha), "threshold": float(threshold)
                    },
                    "mcc": metrics["mcc"],
                    "pr_auc": metrics["pr_auc"],
                    "fpr": metrics["fpr"],
                }
            )
                              
    for classifier_threshold in (0.3, 0.5, 0.7, 0.9):
        for anomaly_threshold in config["selection"]["validation_threshold_quantiles"]:
            prediction = (
                (classifier_select >= classifier_threshold)
                | (anomaly_select >= anomaly_threshold)
            )
            score = np.maximum(
                classifier_select / classifier_threshold,
                anomaly_select / anomaly_threshold,
            )
            score = np.clip(score / 2, 0, 1)
            metrics = binary_metrics(y_select, prediction, score)
            candidates.append(
                {
                    "family": "logical_or",
                    "parameters": {
                        "classifier_threshold": classifier_threshold,
                        "anomaly_threshold": anomaly_threshold,
                    },
                    "mcc": metrics["mcc"],
                    "pr_auc": metrics["pr_auc"],
                    "fpr": metrics["fpr"],
                }
            )
                                                                                 
                                      
    meta_train = np.column_stack(
        [
            classifier_probability[calibration_indices],
            anomaly_probability[calibration_indices],
            _entropy(classifier_probability[calibration_indices]),
        ]
    )
    meta_select = np.column_stack(
        [classifier_select, anomaly_select, _entropy(classifier_select)]
    )
    meta = LogisticRegression(class_weight="balanced", random_state=seed)
    meta.fit(meta_train, y_validation[calibration_indices])
    stack_score = meta.predict_proba(meta_select)[:, 1]
    for threshold in thresholds:
        metrics = binary_metrics(y_select, stack_score >= threshold, stack_score)
        candidates.append(
            {
                "family": "logistic_stacking",
                "parameters": {
                    "threshold": float(threshold),
                    "coefficients": meta.coef_[0].tolist(),
                    "intercept": float(meta.intercept_[0]),
                },
                "mcc": metrics["mcc"],
                "pr_auc": metrics["pr_auc"],
                "fpr": metrics["fpr"],
            }
        )
    candidate_table = pd.DataFrame(
        [
            {
                **{key: value for key, value in candidate.items() if key != "parameters"},
                "parameters_json": json.dumps(candidate["parameters"], sort_keys=True),
            }
            for candidate in candidates
        ]
    ).sort_values(["mcc", "pr_auc"], ascending=False)
    candidate_table.to_csv(paths["tables"] / "fusion_candidate_selection.csv", index=False)
    winner_row = candidate_table.iloc[0]
    winner = {
        "family": str(winner_row["family"]),
        "parameters": json.loads(winner_row["parameters_json"]),
        "selection_mcc": float(winner_row["mcc"]),
        "selection_pr_auc": float(winner_row["pr_auc"]),
        "selection_fpr": float(winner_row["fpr"]),
    }
    freeze = {
        "status": "FROZEN_BEFORE_EXTERNAL_EVALUATION",
        "selection_data": (
            "fixed stratified sample of leakage-audited, multi-family "
            "CICIDS2017 validation partition only"
        ),
        "selection_rows": int(len(y_validation)),
        "seed": seed,
        "calibration_method": calibration_name,
        "anomaly_scaler": scaler_name,
        "anomaly_model": anomaly_name,
        "fusion": winner,
        "external_labels_used_for_selection": False,
    }
    write_json(paths["manifests"] / "method_freeze.json", freeze)
    return freeze


if __name__ == "__main__":
    result = select()
    print(json.dumps(result, indent=2))
