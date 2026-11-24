from __future__ import annotations
import gc
import json
import pickle
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from benchmark.common import resolve, set_seed
from benchmark.metrics import binary_metrics
from benchmark.model_registry import MODEL_NAMES, model_and_space
from benchmark.readiness_anomaly import (
    empirical_probability,
    make_scaler,
    train_anomaly,
)
from benchmark.readiness_supervised import load_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
def config() -> dict:
    with resolve("benchmark/readiness_config.yaml").open() as handle:
        return yaml.safe_load(handle)
def paths() -> dict[str, Path]:
    output = {key: resolve(value) for key, value in config()["paths"].items()}
    for path in output.values():
        path.mkdir(parents=True, exist_ok=True)
    (output["results"] / "predictions").mkdir(parents=True, exist_ok=True)
    return output
def load_external() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from benchmark.supervised import load_partition
    X_cse, y_cse, _ = load_partition("cic_ids2018_external")
    cse_frame = pd.read_csv(
        resolve("data/processed/cic_ids2018/cic2018_aligned_features.csv"),
        usecols=[
            "byte_count", "bytes_per_packet", "log_byte_count", "ack_flag_count",
            "flag_sum", "ack_ratio", "fin_ratio", "syn_ratio", "packet_rate",
            "log_packet_rate", "label",
        ],
        low_memory=False,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    cse_frame["binary"] = (cse_frame["label"] != 0).astype("int8")
    cse_frame.drop_duplicates(
        subset=[
            "byte_count", "bytes_per_packet", "log_byte_count", "ack_flag_count",
            "flag_sum", "ack_ratio", "fin_ratio", "syn_ratio", "packet_rate",
            "log_packet_rate", "binary",
        ],
        inplace=True,
    )
    cse_family = cse_frame["label"].to_numpy(dtype=np.int16)
    if len(cse_family) != len(y_cse):
        raise RuntimeError("CSE family alignment differs from deduplicated protocol")
    unsw = np.load(paths()["data"] / "cic_unsw_nb15_external.npz")
    return {
        "cic_ids2018_external": (X_cse, y_cse, cse_family),
        "cic_unsw_nb15_external": (
            unsw["X"], unsw["y"], unsw["family"].astype(np.int16)
        ),
    }
def _tuned_pipeline(name: str, seed: int, y: np.ndarray):
    tuning = json.loads(
        (paths()["results"] / f"tuning_{name}.json").read_text()
    )
    weight = float((y == 0).sum() / (y == 1).sum())
    model, _ = model_and_space(name, seed, weight)
    pipeline = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    pipeline.set_params(**tuning["best_parameters"])
    if name == "mlp":
        pipeline.set_params(
            model__max_iter=int(config()["selection"]["final_mlp_max_iter"]),
            model__batch_size=int(config()["selection"]["final_mlp_batch_size"]),
        )
    return pipeline
def _probability(model, X: np.ndarray, batch: int = 150000) -> np.ndarray:
    return np.concatenate(
        [np.asarray(model.predict_proba(X[i : i + batch]))[:, 1]
         for i in range(0, len(X), batch)]
    )
def _model_bytes(model, target: Path) -> int:
    with target.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target.stat().st_size
def _evaluate_row(
    seed: int, model_name: str, dataset: str, y: np.ndarray,
    family: np.ndarray, probability: np.ndarray, threshold: float,
    training_seconds: float, inference_seconds: float, model_bytes: int,
    peak_memory_mb: float,
) -> dict:
    prediction = probability >= threshold
    row = {
        "seed": seed, "model": model_name, "dataset": dataset,
        "threshold": threshold, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_us_per_flow": inference_seconds * 1e6 / len(y),
        "model_bytes": model_bytes, "peak_memory_mb": peak_memory_mb,
        **binary_metrics(y, prediction, probability),
    }
    if seed == 42:
        np.savez_compressed(
            paths()["results"] / "predictions" / f"{model_name}_{dataset}_seed_42.npz",
            y_true=y.astype(np.int8), prediction=prediction.astype(np.int8),
            probability=probability.astype(np.float32), family=family,
        )
    return row
def run_supervised() -> pd.DataFrame:
    output = paths()
    checkpoint = output["results"] / "final_supervised_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(int(r["seed"]), r["model"]) for r in rows}
    X_train, y_train, _, _ = load_data("train")
    X_test, y_test, family_test, _ = load_data("test")
    datasets = {
        "cicids2017_internal": (X_test, y_test, family_test),
        **load_external(),
    }
    for seed in config()["study"]["seeds"]:
        for name in MODEL_NAMES:
            if (seed, name) in completed:
                continue
            set_seed(seed)
            model = _tuned_pipeline(name, seed, y_train)
            start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            started = time.perf_counter()
            model.fit(X_train, y_train)
            training_seconds = time.perf_counter() - started
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            model_path = output["models"] / f"{name}_seed_{seed}.pkl"
            model_bytes = _model_bytes(model, model_path)
            for dataset, (X, y, family) in datasets.items():
                started = time.perf_counter()
                probability = _probability(model, X)
                inference_seconds = time.perf_counter() - started
                rows.append(
                    _evaluate_row(
                        seed, name, dataset, y, family, probability, 0.5,
                        training_seconds, inference_seconds, model_bytes,
                        max(0.0, (peak_rss - start_rss) / 1024),
                    )
                )
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"completed supervised seed={seed} model={name}", flush=True)
            del model
            gc.collect()
    return pd.DataFrame(rows)
def run_hybrid() -> pd.DataFrame:
    output = paths()
    checkpoint = output["results"] / "final_hybrid_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {int(r["seed"]) for r in rows}
    frozen = json.loads((output["manifests"] / "method_freeze.json").read_text())
    params = frozen["fusion"]["parameters"]
    coefficients = np.asarray(params["coefficients"], dtype=float)
    intercept = float(params["intercept"])
    threshold = float(params["threshold"])
    X_train, y_train, _, _ = load_data("train")
    X_val, y_val, _, _ = load_data("validation")
    X_test, y_test, family_test, _ = load_data("test")
    datasets = {
        "cicids2017_internal": (X_test, y_test, family_test),
        **load_external(),
    }
    supervised_runs = pd.read_csv(
        output["results"] / "final_supervised_runs.csv"
    )
    for seed in config()["study"]["seeds"]:
        if seed in completed:
            continue
        set_seed(seed)
        start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.perf_counter()
        with (
            output["models"] / f"random_forest_seed_{seed}.pkl"
        ).open("rb") as handle:
            rf = pickle.load(handle)
        rf_training_seconds = float(
            supervised_runs[
                (supervised_runs["seed"] == seed)
                & (supervised_runs["model"] == "random_forest")
            ]["training_seconds"].iloc[0]
        )
        indices = np.arange(len(y_val))
        calibration_indices, _ = train_test_split(
            indices, test_size=0.5, stratify=y_val, random_state=seed
        )
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(rf), method=frozen["calibration_method"]
        ).fit(X_val[calibration_indices], y_val[calibration_indices])
        rng = np.random.default_rng(seed)
        normal = X_train[y_train == 0]
        cap = int(config()["selection"]["anomaly_training_normal_cap"])
        normal_fit = normal[
            rng.choice(len(normal), min(cap, len(normal)), replace=False)
        ]
        scaler = make_scaler(frozen["anomaly_scaler"], seed).fit(normal_fit)
        detector = train_anomaly(
            frozen["anomaly_model"],
            scaler.transform(normal_fit).astype(np.float32),
            seed,
        )
        reference = detector.score(
            scaler.transform(
                X_val[calibration_indices][y_val[calibration_indices] == 0]
            ).astype(np.float32)
        )
        training_seconds = rf_training_seconds + (time.perf_counter() - started)
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        bundle = {
            "classifier": calibrated, "anomaly_scaler": scaler,
            "anomaly_detector": detector, "normal_reference": reference,
            "freeze": frozen,
        }
        model_path = output["models"] / f"hybrid_seed_{seed}.pkl"
        model_bytes = _model_bytes(bundle, model_path)
        for dataset, (X, y, family) in datasets.items():
            started = time.perf_counter()
            classifier_probability = _probability(calibrated, X)
            anomaly_probability = empirical_probability(
                detector.score(scaler.transform(X).astype(np.float32)), reference
            )
            p = np.clip(classifier_probability, 1e-7, 1 - 1e-7)
            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            linear = (
                coefficients[0] * classifier_probability
                + coefficients[1] * anomaly_probability
                + coefficients[2] * entropy
                + intercept
            )
            probability = 1 / (1 + np.exp(-np.clip(linear, -40, 40)))
            inference_seconds = time.perf_counter() - started
            rows.append(
                _evaluate_row(
                    seed, "frozen_hybrid", dataset, y, family, probability,
                    threshold, training_seconds, inference_seconds, model_bytes,
                    max(0.0, (peak_rss - start_rss) / 1024),
                )
            )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(f"completed hybrid seed={seed}", flush=True)
        del rf, calibrated, detector, bundle
        gc.collect()
    return pd.DataFrame(rows)
if __name__ == "__main__":
    run_supervised()
    run_hybrid()
