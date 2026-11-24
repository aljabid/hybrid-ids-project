from __future__ import annotations
import argparse
from functools import partial
import json
import os
import pickle
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from benchmark.common import load_config, resolve, set_seed, write_json
from benchmark.metrics import binary_metrics
from benchmark.model_registry import MODEL_NAMES, model_and_space
def load_partition(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(resolve(f"archive/benchmark_v1/data/{name}.npz"), allow_pickle=False)
    return data["X"], data["y"], data["features"].astype(str).tolist()
class PeakMemoryMonitor:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
    def __enter__(self):
        process = psutil.Process(os.getpid())
        def poll():
            while not self._stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self._stop.wait(self.interval)
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self
    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join()
def _pipeline(
    name: str, seed: int, positive_weight: float, k: int, n_features: int
) -> tuple[Pipeline, dict]:
    model, space = model_and_space(name, seed, positive_weight)
    selector = (
        "passthrough"
        if k >= n_features
        else SelectKBest(
            score_func=partial(mutual_info_classif, random_state=seed), k=k
        )
    )
    pipe = Pipeline(
        [
            ("selector", selector),
            ("scaler", MinMaxScaler()),
            ("model", model),
        ]
    )
    return pipe, space
def tune_models(config_path: str = "benchmark/config.yaml", force: bool = False) -> pd.DataFrame:
    config = load_config(resolve(config_path))
    X, y, features = load_partition("cicids2017_train")
    exp = config["experiments"]
    seed = int(config["data"]["random_seed"])
    set_seed(seed)
    positive_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    max_rows = int(exp["tuning_max_rows"])
    if len(y) > max_rows:
        rng = np.random.default_rng(seed)
        selected = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == label), size=min((y == label).sum(), max_rows // 2), replace=False)
                for label in (0, 1)
            ]
        )
        X_tune, y_tune = X[selected], y[selected]
    else:
        X_tune, y_tune = X, y
    scorer = make_scorer(matthews_corrcoef)
    cv = StratifiedKFold(
        n_splits=int(exp["cv_folds"]), shuffle=True, random_state=seed
    )
    rows = []
    for name in MODEL_NAMES:
        model_path = resolve(f"archive/benchmark_v1/models/{name}_tuned.pkl")
        result_path = resolve(f"archive/benchmark_v1/results/tuning_{name}.json")
        if model_path.exists() and result_path.exists() and not force:
            rows.append(json.loads(result_path.read_text()))
            continue
        pipe, space = _pipeline(
            name,
            seed,
            positive_weight,
            min(int(config["features"]["select_k"]), len(features)),
            len(features),
        )
        search = RandomizedSearchCV(
            pipe,
            param_distributions=space,
            n_iter=int(exp["search_iterations"]),
            scoring=scorer,
            cv=cv,
            random_state=seed,
            n_jobs=int(exp["n_jobs"]),
            verbose=1,
            return_train_score=False,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(X_tune, y_tune)
        elapsed = time.perf_counter() - started
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(search.best_estimator_, handle)
        fitted_selector = search.best_estimator_.named_steps["selector"]
        selected_indices = (
            np.arange(len(features))
            if fitted_selector == "passthrough"
            else fitted_selector.get_support(indices=True)
        )
        row = {
            "model": name,
            "best_cv_mcc": float(search.best_score_),
            "cv_std_at_best": float(search.cv_results_["std_test_score"][search.best_index_]),
            "search_seconds": elapsed,
            "n_candidates": int(len(search.cv_results_["params"])),
            "best_parameters": search.best_params_,
            "selected_features": [features[index] for index in selected_indices],
        }
        write_json(result_path, row)
        rows.append(row)
    table = pd.DataFrame(rows)
    resolve("archive/benchmark_v1/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(resolve("archive/benchmark_v1/tables/hyperparameter_summary.csv"), index=False)
    return table
def _predict_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    score = np.asarray(model.decision_function(X))
    return 1.0 / (1.0 + np.exp(-score))
def run_benchmark(config_path: str = "benchmark/config.yaml", force: bool = False) -> pd.DataFrame:
    config = load_config(resolve(config_path))
    seeds = [int(seed) for seed in config["experiments"]["seeds"]]
    X_train, y_train, features = load_partition("cicids2017_train")
    datasets = {
        "cicids2017_internal": load_partition("cicids2017_test")[:2],
        "cic_ids2018_external": load_partition("cic_ids2018_external")[:2],
    }
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    all_rows = []
    prediction_dir = resolve("archive/benchmark_v1/results/predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        tuned_path = resolve(f"archive/benchmark_v1/results/tuning_{name}.json")
        if not tuned_path.exists():
            raise FileNotFoundError(f"Run tuning first: {tuned_path}")
        tuned = json.loads(tuned_path.read_text())
        for seed in seeds:
            output_model = resolve(f"archive/benchmark_v1/models/{name}_seed_{seed}.pkl")
            set_seed(seed)
            pipe, _ = _pipeline(
                name,
                seed,
                positive_weight,
                min(int(config["features"]["select_k"]), len(features)),
                len(features),
            )
            pipe.set_params(**tuned["best_parameters"])
            with PeakMemoryMonitor() as monitor:
                started = time.perf_counter()
                pipe.fit(X_train, y_train)
                train_seconds = time.perf_counter() - started
            with output_model.open("wb") as handle:
                pickle.dump(pipe, handle)
            model_size = output_model.stat().st_size
            for dataset, (X_test, y_test) in datasets.items():
                _predict_probability(pipe, X_test[: min(1000, len(X_test))])
                started = time.perf_counter()
                probability = _predict_probability(pipe, X_test)
                inference_seconds = time.perf_counter() - started
                prediction = (probability >= 0.5).astype(np.int8)
                metrics = binary_metrics(y_test, prediction, probability)
                row = {
                    "model": name,
                    "seed": seed,
                    "dataset": dataset,
                    **metrics,
                    "training_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "microseconds_per_flow": inference_seconds / len(y_test) * 1e6,
                    "flows_per_second": len(y_test) / max(inference_seconds, 1e-12),
                    "peak_rss_mb": monitor.peak / (1024**2),
                    "model_size_mb": model_size / (1024**2),
                }
                all_rows.append(row)
                np.savez_compressed(
                    prediction_dir / f"{name}_{dataset}_seed_{seed}.npz",
                    y_true=y_test.astype(np.int8),
                    y_pred=prediction,
                    probability=probability.astype(np.float32),
                )
    results = pd.DataFrame(all_rows)
    results.to_csv(resolve("archive/benchmark_v1/results/supervised_seed_results.csv"), index=False)
    metric_columns = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "mcc", "tpr", "fpr", "ece", "brier",
        "negative_log_likelihood", "training_seconds", "inference_seconds",
        "microseconds_per_flow", "flows_per_second", "peak_rss_mb", "model_size_mb",
    ]
    summary = (
        results.groupby(["model", "dataset"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(resolve("archive/benchmark_v1/tables/supervised_benchmark_summary.csv"), index=False)
    return results
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/config.yaml")
    parser.add_argument("--mode", choices=["tune", "benchmark", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode in {"tune", "all"}:
        tune_models(args.config, args.force)
    if args.mode in {"benchmark", "all"}:
        run_benchmark(args.config, args.force)
if __name__ == "__main__":
    main()
