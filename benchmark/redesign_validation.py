                                                                                        

from __future__ import annotations

import gc
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from benchmark.common import set_seed, write_json
from benchmark.metrics import binary_metrics
from benchmark.readiness_anomaly import empirical_probability, make_scaler, train_anomaly
from benchmark.redesign_hybrid import _threshold_search
from benchmark.redesign_preprocessing import config, load_partition, paths
from benchmark.redesign_supervised import (
    MODEL_NAMES, TabularResNet, family_balanced_indices, model_and_space,
    selected_transformer, train_deep,
)


def _external_diagnostics():
    output = paths()
    result = {}
    for name in ("cse_cic_ids2018", "cic_unsw_nb15"):
        data = np.load(
            output["data"] / f"{name}_expanded_diagnostic.npz", allow_pickle=False
        )
        result[name] = (data["X"], data["y"], data["family"])
    return result


def _probability(model, X, batch=100000):
    return np.concatenate([
        model.predict_proba(X[start : start + batch])
        for start in range(0, len(X), batch)
    ])


def architecture_freeze():
    output = paths()
    anomaly = json.loads(
        (output["manifests"] / "anomaly_selection.json").read_text()
    )["winner"]
    fusion = json.loads(
        (output["manifests"] / "fusion_selection.json").read_text()
    )["winner"]
    frozen = {
        "status": "FROZEN_BEFORE_INTERNAL_TEST_AND_NEW_CONFIRMATION",
        "feature_schema": "68 mapped CICFlowMeter features",
        "preprocessing": "winsor_0.1_99.9_log14_robust_5_95",
        "classifier": "family-balanced multiclass Random Forest",
        "classifier_hyperparameters": json.loads(
            (output["results"] / "group_tuning_random_forest.json").read_text()
        )["best_parameters"],
        "calibration": "isotonic on validation calibration quarter",
        "anomaly_scaler": anomaly["scaler"],
        "anomaly_model": anomaly["anomaly_model"],
        "meta_gate": "logistic regression",
        "meta_C": float(fusion.get("meta_C") or 1.0),
        "meta_threshold": float(fusion["classifier_threshold"]),
        "meta_features": [
            "classifier_probability", "anomaly_probability",
            "classifier_entropy", "absolute_disagreement", "probability_product",
        ],
        "selection_sources": "CICIDS2017 train and validation groups only",
        "external_labels_used": False,
        "confirmation_dataset_accessed": False,
    }
    write_json(output["manifests"] / "architecture_freeze.json", frozen)
    return frozen


def _meta_features(classifier_probability, anomaly_probability):
    cp, ap = classifier_probability, anomaly_probability
    entropy = -(
        np.clip(cp, 1e-7, 1) * np.log(np.clip(cp, 1e-7, 1))
        + np.clip(1 - cp, 1e-7, 1) * np.log(np.clip(1 - cp, 1e-7, 1))
    )
    return np.column_stack([cp, ap, entropy, np.abs(cp - ap), cp * ap])


def _evaluate(rows, seed, model, dataset, y, family, probability, threshold,
              training_seconds, inference_seconds):
    prediction = probability >= threshold
    rows.append({
        "seed": seed, "model": model, "dataset": dataset,
        "threshold": threshold, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        **binary_metrics(y, prediction, probability),
    })
    if seed == 42:
        target = paths()["results"] / "predictions"
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / f"{model}_{dataset}_seed_42.npz",
            y_true=y.astype(np.int8), prediction=prediction.astype(np.int8),
            probability=probability.astype(np.float32), family=family,
        )


def run_classical(force=False):
    output = paths()
    architecture = architecture_freeze()
    checkpoint = output["results"] / "grouped_ten_seed_runs.csv"
    if force and checkpoint.exists():
        checkpoint.rename(checkpoint.with_suffix(".pre_force_backup.csv"))
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(int(row["seed"]), row["model"]) for row in rows}
    X, y, family, _, _ = load_partition("train")
    Xv, yv, _, _, _ = load_partition("validation")
    Xt, yt, family_test, _, _ = load_partition("internal_test")
    datasets = {
        "cicids2017_group_test": (Xt, yt, family_test),
        **_external_diagnostics(),
    }
    tuning = {
        name: json.loads(
            (output["results"] / f"group_tuning_{name}.json").read_text()
        ) for name in MODEL_NAMES
    }
    for seed in config()["study"]["seeds"]:
        index = family_balanced_indices(family, y, seed)
        for name in MODEL_NAMES:
            if (seed, name) in completed:
                continue
            set_seed(seed)
            estimator, _ = model_and_space(name, seed)
            pipeline = Pipeline([
                ("preprocessor", selected_transformer(seed)),
                ("model", estimator),
            ])
            pipeline.set_params(**tuning[name]["best_parameters"])
            started = time.perf_counter()
            pipeline.fit(X[index], y[index])
            training = time.perf_counter() - started
            for dataset, (Xd, yd, fd) in datasets.items():
                started = time.perf_counter()
                probability = _probability(pipeline, Xd)[:, 1]
                inference = time.perf_counter() - started
                _evaluate(
                    rows, seed, name, dataset, yd, fd, probability, 0.5,
                    training, inference,
                )
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"validated seed={seed} model={name}", flush=True)
            del pipeline
            gc.collect()
        if (seed, "family_aware_rf") not in completed:
            estimator, _ = model_and_space("random_forest", seed)
            classifier = Pipeline([
                ("preprocessor", selected_transformer(seed)),
                ("model", estimator),
            ])
            classifier.set_params(**tuning["random_forest"]["best_parameters"])
            started = time.perf_counter()
            classifier.fit(X[index], family[index])
            classifier_training = time.perf_counter() - started
            benign_class = int(np.unique(family[y == 0])[0])
            classes = list(classifier.classes_)
            val_raw = _probability(classifier, Xv)
            val_attack = 1 - val_raw[:, classes.index(benign_class)]
            rng = np.random.default_rng(seed)
            development, _ = train_test_indices(yv, seed)
            calibration, meta = split_indices(development, yv, seed)
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                val_attack[calibration], yv[calibration]
            )
            normal = X[y == 0]
            normal_fit = normal[
                rng.choice(
                    len(normal), config()["selection"]["anomaly_normal_rows"],
                    replace=False,
                )
            ]
            anomaly_scaler = make_scaler(architecture["anomaly_scaler"], seed).fit(normal_fit)
            anomaly_detector = train_anomaly(
                architecture["anomaly_model"],
                anomaly_scaler.transform(normal_fit).astype(np.float32),
                seed, epochs=20,
            )
            val_scaled = anomaly_scaler.transform(Xv).astype(np.float32)
            reference = anomaly_detector.score(
                val_scaled[calibration][yv[calibration] == 0]
            )
            val_anomaly = empirical_probability(
                anomaly_detector.score(val_scaled), reference
            )
            val_calibrated = calibrator.transform(val_attack)
            meta_model = LogisticRegression(
                C=architecture["meta_C"], class_weight="balanced",
                random_state=seed, max_iter=1000,
            ).fit(_meta_features(val_calibrated[meta], val_anomaly[meta]), yv[meta])
            hybrid_training = time.perf_counter() - started
            for dataset, (Xd, yd, fd) in datasets.items():
                started_inference = time.perf_counter()
                raw = _probability(classifier, Xd)
                classifier_probability = calibrator.transform(
                    1 - raw[:, classes.index(benign_class)]
                )
                anomaly_probability = empirical_probability(
                    anomaly_detector.score(
                        anomaly_scaler.transform(Xd).astype(np.float32)
                    ),
                    reference,
                )
                meta_probability = meta_model.predict_proba(
                    _meta_features(classifier_probability, anomaly_probability)
                )[:, 1]
                inference = time.perf_counter() - started_inference
                _evaluate(
                    rows, seed, "family_aware_rf", dataset, yd, fd,
                    classifier_probability, 0.80, classifier_training, inference,
                )
                _evaluate(
                    rows, seed, "redesigned_meta_hybrid", dataset, yd, fd,
                    meta_probability, architecture["meta_threshold"],
                    hybrid_training, inference,
                )
            with (output["models"] / f"redesigned_hybrid_seed_{seed}.pkl").open("wb") as handle:
                pickle.dump({
                    "classifier": classifier, "calibrator": calibrator,
                    "anomaly_scaler": anomaly_scaler,
                    "anomaly_detector": anomaly_detector,
                    "anomaly_reference": reference, "meta_model": meta_model,
                    "architecture": architecture,
                }, handle)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"validated seed={seed} family-aware + hybrid", flush=True)
            gc.collect()
    return pd.DataFrame(rows)


def train_test_indices(y, seed):
    from sklearn.model_selection import train_test_split
    development, selection = train_test_split(
        np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed
    )
    return development, selection


def split_indices(indices, y, seed):
    from sklearn.model_selection import train_test_split
    calibration, meta = train_test_split(
        indices, test_size=0.5, stratify=y[indices], random_state=seed
    )
    return calibration, meta


def run_deep():
    output = paths()
    checkpoint = output["results"] / "grouped_ten_seed_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records")
    completed = {(int(row["seed"]), row["model"]) for row in rows}
    datasets = {
        "cicids2017_group_test": (
            load_partition("internal_test")[0],
            load_partition("internal_test")[1],
            load_partition("internal_test")[2],
        ),
        **_external_diagnostics(),
    }
    for seed in config()["study"]["seeds"]:
        if (seed, "tabular_resnet") in completed:
            continue
        started = time.perf_counter()
        history = train_deep(seed)
        training = time.perf_counter() - started
        bundle = torch.load(
            output["models"] / f"tabular_resnet_seed_{seed}.pt",
            map_location="cpu", weights_only=True,
        )
        model = TabularResNet(bundle["input_dim"])
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        with (output["models"] / f"tabular_resnet_preprocessor_seed_{seed}.pkl").open("rb") as handle:
            preprocessor = pickle.load(handle)
        for dataset, (Xd, yd, fd) in datasets.items():
            started_inference = time.perf_counter()
            probabilities = []
            with torch.no_grad():
                for start in range(0, len(Xd), 50000):
                    transformed = preprocessor.transform(Xd[start : start + 50000])
                    probabilities.append(
                        torch.sigmoid(model(torch.as_tensor(transformed))).numpy()
                    )
            probability = np.concatenate(probabilities)
            _evaluate(
                rows, seed, "tabular_resnet", dataset, yd, fd, probability,
                0.5, training, time.perf_counter() - started_inference,
            )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(f"validated seed={seed} tabular_resnet", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_classical()
    run_deep()
