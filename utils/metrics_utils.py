import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from utils.logger import get_logger
logger = get_logger(__name__)
def compute_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))
def compute_precision(y_true, y_pred, average='weighted') -> float:
    return float(precision_score(y_true, y_pred, average=average,
                                 zero_division=0))
def compute_recall(y_true, y_pred, average='weighted') -> float:
    return float(recall_score(y_true, y_pred, average=average,
                              zero_division=0))
def compute_f1(y_true, y_pred, average='weighted') -> float:
    return float(f1_score(y_true, y_pred, average=average,
                          zero_division=0))
def compute_roc_auc(y_true, y_score, multi_class=False) -> float:
    try:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_score))
        else:
            return float(roc_auc_score(y_true, y_score,
                                       multi_class='ovr',
                                       average='weighted'))
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        return 0.0
def compute_tpr_fpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        tpr_list, fpr_list = [], []
        n_classes = cm.shape[0]
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        tpr = float(np.mean(tpr_list))
        fpr = float(np.mean(fpr_list))
    return float(tpr), float(fpr)
def compute_classification_report(y_true, y_pred, target_names=None) -> dict:
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    return report
def compute_ece(y_true, y_pred, confidence_scores, n_bins=10) -> tuple:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence_scores = np.asarray(confidence_scores, dtype=float)
    if set(np.unique(y_true)).issubset({0, 1}) and set(np.unique(y_pred)).issubset({0, 1}):
        confidence_scores = np.where(
            y_pred == 1, confidence_scores, 1.0 - confidence_scores
        )
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    bin_data = []
    for i in range(n_bins):
        in_bin = (confidence_scores >= bins[i]) & (
            confidence_scores <= bins[i + 1]
            if i == n_bins - 1
            else confidence_scores < bins[i + 1]
        )
        bin_size = int(in_bin.sum())
        if bin_size == 0:
            continue
        bin_acc  = float((y_true[in_bin] == y_pred[in_bin]).mean())
        bin_conf = float(confidence_scores[in_bin].mean())
        weight   = bin_size / n_total
        ece += weight * abs(bin_acc - bin_conf)
        bin_data.append({
            "bin":        f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            "size":       bin_size,
            "accuracy":   round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
            "gap":        round(abs(bin_acc - bin_conf), 4)
        })
    return float(ece), bin_data
def full_metrics_dict(y_true, y_pred, dataset_name="") -> dict:
    tpr, fpr = compute_tpr_fpr(y_true, y_pred)
    return {
        "dataset":    dataset_name,
        "n_samples":  int(len(y_true)),
        "accuracy":   round(compute_accuracy(y_true, y_pred), 4),
        "precision":  round(compute_precision(y_true, y_pred), 4),
        "recall":     round(compute_recall(y_true, y_pred), 4),
        "f1":         round(compute_f1(y_true, y_pred), 4),
        "tpr":        round(tpr, 4),
        "fpr":        round(fpr, 4),
    }
