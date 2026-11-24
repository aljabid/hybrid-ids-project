import numpy as np
from utils.logger import get_logger
logger = get_logger(__name__)
def compute_shap_values(clf, X_background, X_explain):
    try:
        import shap
    except ImportError:
        logger.error("shap not installed — run: pip install shap")
        raise
    logger.info(f"Computing SHAP values for {X_explain.shape[0]} samples")
    explainer   = shap.TreeExplainer(clf, X_background)
    shap_values = explainer.shap_values(X_explain)
    logger.info("SHAP computation complete")
    return explainer, shap_values
def mean_abs_shap_per_class(shap_values, class_names, feature_names):
    results = {}
    for i, cls in enumerate(class_names):
        try:
            if isinstance(shap_values, list):
                vals = np.array(shap_values[i])
            else:
                vals = np.array(shap_values)
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)
            mean_abs = np.abs(vals).mean(axis=0)
            if mean_abs.ndim == 0:
                mean_abs = np.array([float(mean_abs)])
            results[cls] = {feat: round(float(v), 5)
                            for feat, v in zip(feature_names, mean_abs)}
        except Exception as e:
            results[cls] = {feat: 0.0 for feat in feature_names}
    return results
def top_features_per_class(shap_class_dict, top_n=3):
    summary = {}
    for cls, feat_dict in shap_class_dict.items():
        sorted_feats = sorted(feat_dict.items(),
                               key=lambda x: x[1], reverse=True)
        summary[cls] = sorted_feats[:top_n]
    return summary
