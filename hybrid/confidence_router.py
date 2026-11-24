import numpy as np
import torch
import yaml
from utils.confidence_utils import get_confidence, format_routing_decision
from utils.logger import get_logger
from utils.feature_config import LABEL_MAP_LOCAL_INV
logger = get_logger(__name__)
def load_threshold(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['hybrid']['confidence_threshold']
def reconstruction_error(autoencoder, x_array):
    x_tensor = torch.FloatTensor(x_array.reshape(1, -1))
    with torch.no_grad():
        err = autoencoder.reconstruction_error(x_tensor)
    return float(err.item())
def route(x, clf, autoencoder, ae_threshold,
          confidence_threshold=None, config_path="config/config.yaml"):
    if confidence_threshold is None:
        confidence_threshold = load_threshold(config_path)
    pred_label, confidence = get_confidence(clf, x)
    pred_label_str = LABEL_MAP_LOCAL_INV.get(int(pred_label), str(pred_label))
    meta = {
        "classifier_label":      pred_label_str,
        "confidence":            round(float(confidence), 4),
        "confidence_threshold":  confidence_threshold,
        "ae_error":              None,
        "ae_threshold":          ae_threshold,
        "module":                None,
        "final_label":           None
    }
    if confidence >= confidence_threshold and pred_label_str != "Normal":
        meta["module"]       = "classifier"
        meta["final_label"]  = pred_label_str
        logger.debug(format_routing_decision(
            pred_label_str, confidence, confidence_threshold, "classifier"))
        return pred_label_str, "classifier", meta
    error = reconstruction_error(autoencoder, x)
    meta["ae_error"] = round(error, 6)
    if error > ae_threshold:
        meta["module"]      = "autoencoder"
        meta["final_label"] = "ANOMALY"
        logger.debug(format_routing_decision(
            "ANOMALY", confidence, confidence_threshold, "autoencoder"))
        return "ANOMALY", "autoencoder", meta
    else:
        meta["module"]      = "autoencoder"
        meta["final_label"] = "Normal"
        logger.debug(format_routing_decision(
            "Normal", confidence, confidence_threshold, "autoencoder"))
        return "Normal", "autoencoder", meta
def route_batch(X, clf, autoencoder, ae_threshold,
                confidence_threshold=None, config_path="config/config.yaml"):
    if confidence_threshold is None:
        confidence_threshold = load_threshold(config_path)
    proba      = clf.predict_proba(X)
    pred_int   = clf.predict(X)
    confidence = np.max(proba, axis=1)
    X_tensor  = torch.FloatTensor(X)
    with torch.no_grad():
        ae_errors = autoencoder.reconstruction_error(X_tensor).numpy()
    labels, modules, metas = [], [], []
    for i in range(len(X)):
        pred_str = LABEL_MAP_LOCAL_INV.get(int(pred_int[i]), str(pred_int[i]))
        conf     = float(confidence[i])
        err      = float(ae_errors[i])
        meta = {
            "classifier_label":     pred_str,
            "confidence":           round(conf, 4),
            "confidence_threshold": confidence_threshold,
            "ae_error":             round(err, 6),
            "ae_threshold":         ae_threshold,
        }
        if conf >= confidence_threshold and pred_str != "Normal":
            label  = pred_str
            module = "classifier"
        elif err > ae_threshold:
            label  = "ANOMALY"
            module = "autoencoder"
        else:
            label  = "Normal"
            module = "autoencoder"
        meta["module"]      = module
        meta["final_label"] = label
        labels.append(label)
        modules.append(module)
        metas.append(meta)
    return labels, modules, metas
