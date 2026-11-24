from __future__ import annotations
import json
import pickle
import time
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler,
)
from benchmark.common import resolve, set_seed, write_json
from benchmark.metrics import binary_metrics
def config():
    with resolve("benchmark/redesign_config.yaml").open() as handle:
        return yaml.safe_load(handle)
def paths():
    result = {key: resolve(value) for key, value in config()["paths"].items()}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result
def load_partition(name: str):
    data = np.load(
        paths()["data"] / f"cicids2017_expanded_{name}.npz", allow_pickle=False
    )
    return data["X"], data["y"], data["family"], data["group"], data["features"].astype(str).tolist()
class RobustFlowTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, method: str = "robust", log_indices: tuple[int, ...] = (),
        clip_quantiles: tuple[float, float] = (0.001, 0.999), random_state: int = 42,
    ):
        self.method = method
        self.log_indices = log_indices
        self.clip_quantiles = clip_quantiles
        self.random_state = random_state
    def _log(self, X):
        transformed = np.asarray(X, dtype=np.float64).copy()
        if self.log_indices:
            transformed[:, self.log_indices] = np.log1p(
                np.clip(transformed[:, self.log_indices], 0, None)
            )
        return transformed
    def fit(self, X, y=None):
        transformed = self._log(X)
        self.lower_ = np.quantile(transformed, self.clip_quantiles[0], axis=0)
        self.upper_ = np.quantile(transformed, self.clip_quantiles[1], axis=0)
        clipped = np.clip(transformed, self.lower_, self.upper_)
        if self.method == "minmax":
            self.scaler_ = MinMaxScaler()
        elif self.method == "standard":
            self.scaler_ = StandardScaler()
        elif self.method == "robust":
            self.scaler_ = RobustScaler(quantile_range=(5, 95))
        elif self.method == "quantile":
            self.scaler_ = QuantileTransformer(
                n_quantiles=1000, output_distribution="normal",
                subsample=100000, random_state=self.random_state,
            )
        else:
            raise KeyError(self.method)
        self.scaler_.fit(clipped)
        return self
    def transform(self, X):
        transformed = self._log(X)
        return self.scaler_.transform(
            np.clip(transformed, self.lower_, self.upper_)
        ).astype(np.float32)
def stratified_sample(y, cap, seed):
    rng = np.random.default_rng(seed)
    per = cap // 2
    selected = np.concatenate([
        rng.choice(np.flatnonzero(y == label), min(per, int((y == label).sum())), replace=False)
        for label in (0, 1)
    ])
    rng.shuffle(selected)
    return selected
def benchmark(seed: int = 42, force: bool = False) -> pd.DataFrame:
    output = paths()
    table_path = output["tables"] / "preprocessing_selection.csv"
    if table_path.exists() and not force:
        return pd.read_csv(table_path)
    X_train, y_train, _, _, features = load_partition("train")
    X_val, y_val, _, _, _ = load_partition("validation")
    train_idx = stratified_sample(
        y_train, config()["selection"]["tuning_rows"], seed
    )
    val_idx = stratified_sample(
        y_val, config()["selection"]["validation_rows"], seed
    )
    log_indices = tuple(
        features.index(name) for name in config()["data"]["log1p_features"]
        if name in features
    )
    candidates = [
        ("minmax_raw", "minmax", ()),
        ("winsor_standard_log", "standard", log_indices),
        ("winsor_robust_log", "robust", log_indices),
        ("winsor_quantile_log", "quantile", log_indices),
        ("winsor_robust_raw", "robust", ()),
    ]
    rows = []
    for label, method, logs in candidates:
        set_seed(seed)
        transformer = RobustFlowTransformer(
            method=method, log_indices=logs, random_state=seed
        )
        started = time.perf_counter()
        X_fit = transformer.fit_transform(X_train[train_idx])
        model = RandomForestClassifier(
            n_estimators=50, max_depth=18, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        ).fit(X_fit, y_train[train_idx])
        training_seconds = time.perf_counter() - started
        probability = model.predict_proba(transformer.transform(X_val[val_idx]))[:, 1]
        metrics = binary_metrics(y_val[val_idx], probability >= 0.5, probability)
        rows.append({
            "candidate": label, "method": method, "log_features": len(logs),
            "training_seconds": training_seconds, **metrics,
        })
        with (output["models"] / f"preprocessing_pilot_{label}.pkl").open("wb") as handle:
            pickle.dump({"transformer": transformer, "model": model}, handle)
        print(f"completed preprocessing {label}: MCC={metrics['mcc']:.4f}", flush=True)
    table = pd.DataFrame(rows).sort_values(["mcc", "pr_auc"], ascending=False)
    table.to_csv(table_path, index=False)
    winner = table.iloc[0].to_dict()
    write_json(output["manifests"] / "preprocessing_selection.json", {
        "selection_domain": "CICIDS2017 held-out validation collection groups only",
        "external_labels_used": False, "winner": winner,
    })
    return table
if __name__ == "__main__":
    print(benchmark().to_string(index=False))
