                                                                                

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
                                                                                
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    prediction = (probability >= 0.5).astype(int)
    confidence = np.where(prediction == 1, probability, 1.0 - probability)
    correctness = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    details = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        if not mask.any():
            continue
        accuracy = float(correctness[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        weight = float(mask.mean())
        gap = abs(accuracy - mean_confidence)
        ece += weight * gap
        details.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": int(mask.sum()),
                "accuracy": accuracy,
                "confidence": mean_confidence,
                "gap": gap,
            }
        )
    return float(ece), details


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ece, _ = expected_calibration_error(y_true, probability)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "ece": ece,
        "brier": float(brier_score_loss(y_true, probability)),
        "negative_log_likelihood": float(
            log_loss(y_true, np.column_stack([1 - probability, probability]), labels=[0, 1])
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

