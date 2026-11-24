import pandas as pd
import pickle
from utils.feature_config import FEATURE_NAMES
from utils.logger import get_logger
logger = get_logger(__name__)
MI_FEATURES = ['byte_count', 'bytes_per_packet', 'log_byte_count']
CANONICAL_FEATURES = FEATURE_NAMES
DATASET_REGISTRY = {
    "local": {
        "X_test":     "data/splits/local/X_test.csv",
        "y_test":     "data/splits/local/y_test.csv",
        "metrics_out":"results/metrics/hybrid_metrics_local.json",
        "cm_out":     "results/figures/confusion_matrix_local.png",
        "labels":     ["Normal", "SYN Scan", "TCP Connect", "Xmas Scan"],
        "binary":     False
    },
    "cicids2017": {
        "X_test":     "data/splits/cicids2017/X_test_cicids.csv",
        "y_test":     "data/splits/cicids2017/y_test_cicids.csv",
        "metrics_out":"results/metrics/hybrid_metrics_cicids.json",
        "cm_out":     "results/figures/confusion_matrix_cicids.png",
        "labels":     ["Normal", "PortScan"],
        "binary":     True
    },
    "cic_ids2018": {
        "X_test":     "data/splits/cic_ids2018/X_test_cic2018.csv",
        "y_test":     "data/splits/cic_ids2018/y_test_cic2018.csv",
        "metrics_out":"results/metrics/hybrid_metrics_cic2018.json",
        "cm_out":     "results/figures/confusion_matrix_cic2018.png",
        "labels":     ["Normal", "PortScan"],
        "binary":     True
    }
}
def load_dataset(dataset_name: str):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(DATASET_REGISTRY.keys())}")
    entry = DATASET_REGISTRY[dataset_name]
    X = pd.read_csv(entry['X_test']).values
    y = pd.read_csv(entry['y_test']).values.ravel()
    logger.info(f"Loaded '{dataset_name}': {X.shape[0]} samples, "
                f"{X.shape[1]} features")
    return X, y, entry
def load_scaler(scaler_path="models/classifier/saved/scaler.pkl"):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {scaler_path}")
    return scaler
def load_classifier(model_path="models/classifier/saved/decision_tree_model.pkl"):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    logger.info(f"Classifier loaded from {model_path}")
    return clf
def load_autoencoder(weights_path="models/autoencoder/saved/autoencoder_weights.pth",
                     threshold_path="models/autoencoder/saved/threshold.json"):
    import torch, json
    from models.autoencoder.autoencoder_model import Autoencoder
    ae = Autoencoder()
    ae.load_state_dict(torch.load(weights_path, map_location='cpu'))
    ae.eval()
    with open(threshold_path) as f:
        threshold_data = json.load(f)
    logger.info(f"Autoencoder loaded | threshold = {threshold_data['threshold']:.6f}")
    return ae, threshold_data['threshold']
