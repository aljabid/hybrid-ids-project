                                                                             

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from benchmark.metrics import binary_metrics
from benchmark.readiness_anomaly import (
    empirical_probability, make_scaler, select_threshold, train_anomaly,
)
from benchmark.readiness_final import config, load_external, paths
from benchmark.readiness_supervised import load_data


def _sample(X, y, seed, cap=50000):
    rng = np.random.default_rng(seed)
    per = cap // 2
    index = np.concatenate([
        rng.choice(np.flatnonzero(y == label), min(per, int((y == label).sum())), replace=False)
        for label in (0, 1)
    ])
    rng.shuffle(index)
    return X[index], y[index]


def run() -> pd.DataFrame:
    output = paths()
    checkpoint = output["results"] / "final_anomaly_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(int(r["seed"]), r["anomaly_model"]) for r in rows}
    X_train, y_train, _, _ = load_data("train")
    X_val, y_val, _, _ = load_data("validation")
    X_test, y_test, _, _ = load_data("test")
    external = load_external()
    detectors = config()["anomaly_models"]
    for seed in config()["study"]["seeds"]:
        rng = np.random.default_rng(seed)
        normal = X_train[y_train == 0]
        normal_fit = normal[rng.choice(len(normal), 30000, replace=False)]
        scaler = make_scaler("quantile", seed).fit(normal_fit)
        train_scaled = scaler.transform(normal_fit).astype(np.float32)
        Xv, yv = _sample(X_val, y_val, seed)
        validation_scaled = scaler.transform(Xv).astype(np.float32)
        reference = validation_scaled[yv == 0]
        evaluation = {
            "cicids2017_internal": _sample(X_test, y_test, seed),
            **{
                name: _sample(X, y, seed)
                for name, (X, y, _) in external.items()
            },
        }
        for name in detectors:
            if (seed, name) in completed:
                continue
            detector = train_anomaly(name, train_scaled, seed)
            normal_scores = detector.score(reference)
            val_probability = empirical_probability(
                detector.score(validation_scaled), normal_scores
            )
            threshold, _ = select_threshold(
                yv, val_probability,
                config()["selection"]["validation_threshold_quantiles"],
            )
            for dataset, (X_eval, y_eval) in evaluation.items():
                score = detector.score(scaler.transform(X_eval).astype(np.float32))
                probability = empirical_probability(score, normal_scores)
                rows.append({
                    "seed": seed, "anomaly_model": name, "dataset": dataset,
                    "threshold": threshold, "evaluation_rows": len(y_eval),
                    **binary_metrics(y_eval, probability >= threshold, probability),
                })
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"completed anomaly seed={seed} model={name}", flush=True)
    result = pd.DataFrame(rows)
    summary = result.groupby(["dataset", "anomaly_model"])[
        ["mcc", "pr_auc", "tpr", "fpr", "ece", "brier"]
    ].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(
        output["tables"] / "anomaly_baseline_mean_std.csv", index=False
    )
    return result


if __name__ == "__main__":
    run()
