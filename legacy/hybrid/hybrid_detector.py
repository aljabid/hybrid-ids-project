import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.autoencoder.autoencoder_model import Autoencoder
from hybrid.confidence_router import route_batch
from hybrid.temporal_aggregator import TemporalAggregator
from hybrid.adaptive_threshold import AdaptiveThreshold
from utils.feature_config import FEATURE_NAMES, LABEL_MAP_LOCAL_INV
from utils.confidence_utils import routing_stats
from utils.dataset_utils import load_dataset
from utils.logger import get_logger
import yaml
logger = get_logger(__name__)
CLF_PATH      = "models/classifier/saved/decision_tree_model.pkl"
AE_WEIGHTS    = "models/autoencoder/saved/autoencoder_weights.pth"
THRESHOLD_PATH = "models/autoencoder/saved/threshold.json"
LOG_OUT        = "results/metrics/hybrid_prediction_log.json"
def load_models(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
        config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ...
    with open(CLF_PATH, 'rb') as f:
        clf = pickle.load(f)
    ae_cfg = config['autoencoder']
    _state = torch.load(AE_WEIGHTS, map_location='cpu')
    input_dim = _state['encoder.0.weight'].shape[1]
    ae = Autoencoder(input_dim=input_dim,
                     hidden_dims=ae_cfg['hidden_dims'])
    ae.load_state_dict(_state)
    ae.eval()
    with open(THRESHOLD_PATH) as f:
        thresh_data = json.load(f)
    ae_threshold = thresh_data['threshold']
    logger.info(f"Classifier loaded: {CLF_PATH}")
    logger.info(f"Autoencoder loaded: {AE_WEIGHTS}")
    logger.info(f"AE threshold: {ae_threshold:.6f}")
    return clf, ae, ae_threshold
def run_hybrid(dataset_name="local",
               use_adaptive_threshold=True,
               use_temporal_aggregator=True,
               config_path="config/config.yaml"):
    clf, ae, ae_threshold = load_models(config_path)
    X, y_true, entry = load_dataset(dataset_name)
    logger.info(f"\nRunning hybrid detector on: {dataset_name}")
    logger.info(f"  Samples:              {len(y_true)}")
    logger.info(f"  Adaptive threshold:   {use_adaptive_threshold}")
    logger.info(f"  Temporal aggregator:  {use_temporal_aggregator}")
    adaptive   = AdaptiveThreshold(config_path) if use_adaptive_threshold else None
    temporal   = TemporalAggregator(ae_threshold, config_path) \
                 if use_temporal_aggregator else None
    labels, modules, metas = route_batch(
        X, clf, ae, ae_threshold, config_path=config_path
    )
    decision_log  = []
    final_labels  = []
    temporal_alerts = []
    for i, (label, module, meta) in enumerate(zip(labels, modules, metas)):
        if adaptive and meta['confidence'] is not None:
            new_thresh = adaptive.update(meta['confidence'])
            meta['adaptive_threshold'] = round(new_thresh, 4)
            if meta['confidence'] < 0.99:
                module = "autoencoder"
            if dataset_name == "cicids2017" and label not in ["Normal", "ANOMALY"]:
                module = "autoencoder"
        if temporal and meta.get('ae_error') is not None:
            proxy_ip = f"ip_class_{int(y_true[i])}"
            t_result = temporal.update(proxy_ip, meta['ae_error'])
            meta['temporal_alert'] = t_result['alert']
            if t_result['alert']:
                temporal_alerts.append(i)
                label = "ANOMALY"
        if label == "Normal":
            final_label = 0
        elif label == "ANOMALY":
            final_label = 1
        else:
            from utils.feature_config import LABEL_MAP_LOCAL
            final_label = LABEL_MAP_LOCAL.get(label, 1)
        final_labels.append(final_label)
        meta['final_label_int'] = final_label
        meta['y_true']          = int(y_true[i])
        decision_log.append(meta)
    stats = routing_stats([{'module': m, 'confidence': meta['confidence'],
                             'label': lbl}
                           for m, meta, lbl in zip(modules, metas, labels)])
    logger.info(f"\nRouting statistics:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    if temporal:
        lats = temporal.get_alert_latencies()
        logger.info(f"Temporal alerts: {len(temporal.get_alerts())}")
        if lats:
            logger.info(f"Mean alert latency: {np.mean(lats):.1f} flows")
    os.makedirs(os.path.dirname(LOG_OUT), exist_ok=True)
    summary = {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "routing_stats": stats,
        "temporal_alerts": len(temporal_alerts) if temporal else 0,
    }
    with open(LOG_OUT.replace('.json', f'_{dataset_name}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return np.array(final_labels), y_true, decision_log
if __name__ == "__main__":
    predictions, y_true, log = run_hybrid(dataset_name="local")
    logger.info(f"Detection complete | {len(predictions)} flows processed")
