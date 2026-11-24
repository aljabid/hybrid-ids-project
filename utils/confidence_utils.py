import numpy as np
from utils.logger import get_logger
logger = get_logger(__name__)
def get_confidence(clf, X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    proba       = clf.predict_proba(X)
    labels      = clf.predict(X)
    confidences = np.max(proba, axis=1)
    if confidences.shape[0] == 1:
        return labels[0], float(confidences[0])
    return labels, confidences
def format_routing_decision(label, confidence, threshold, module):
    decision = "DIRECT OUTPUT" if module == "classifier" else "→ AUTOENCODER"
    return (f"Label={label} | Conf={confidence:.3f} "
            f"(thresh={threshold}) | {decision}")
def routing_stats(decisions_log: list) -> dict:
    total = len(decisions_log)
    if total == 0:
        return {}
    n_classifier  = sum(1 for d in decisions_log if d['module'] == 'classifier')
    n_autoencoder = sum(1 for d in decisions_log if d['module'] == 'autoencoder')
    all_conf       = [d['confidence'] for d in decisions_log]
    return {
        "total_flows":                total,
        "handled_by_classifier":      n_classifier,
        "handled_by_autoencoder":     n_autoencoder,
        "pct_classifier":             round(n_classifier / total, 4),
        "pct_autoencoder":            round(n_autoencoder / total, 4),
        "mean_confidence":            round(float(np.mean(all_conf)), 4),
        "std_confidence":             round(float(np.std(all_conf)), 4),
    }
