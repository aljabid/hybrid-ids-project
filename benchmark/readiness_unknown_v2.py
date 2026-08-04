                                                                               

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from benchmark.common import resolve, set_seed
from benchmark.metrics import binary_metrics
from benchmark.readiness_anomaly import empirical_probability, make_scaler, train_anomaly
from benchmark.readiness_final import config, paths
from benchmark.readiness_supervised import load_data


def run() -> pd.DataFrame:
    output = paths()
    checkpoint = output["results"] / "unknown_family_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(int(r["seed"]), r["held_out_family"]) for r in rows}
    parts = [load_data(name) for name in ("train", "validation", "test")]
    X = np.concatenate([part[0] for part in parts])
    y = np.concatenate([part[1] for part in parts])
    family = np.concatenate([part[2] for part in parts]).astype(str)
    counts = pd.Series(family[y == 1]).value_counts()
    held_out_families = counts[counts >= 5000].head(5).index.tolist()
    frozen = json.loads((output["manifests"] / "method_freeze.json").read_text())
    fusion = frozen["fusion"]["parameters"]
    coef = np.asarray(fusion["coefficients"])
    intercept = float(fusion["intercept"])
    threshold = float(fusion["threshold"])
    for seed in config()["study"]["seeds"]:
        rng = np.random.default_rng(seed)
        normal = np.flatnonzero(y == 0)
        for held_out in held_out_families:
            key = (seed, held_out)
            if key in completed:
                continue
            set_seed(seed)
            held = np.flatnonzero(family == held_out)
            known = np.flatnonzero((y == 1) & (family != held_out))
            normal_perm = rng.permutation(normal)
            known_perm = rng.permutation(known)
            normal_train, normal_cal, normal_test = (
                normal_perm[:30000], normal_perm[30000:40000], normal_perm[40000:60000]
            )
            known_train, known_cal = known_perm[:30000], known_perm[30000:40000]
            held_test = rng.choice(held, min(20000, len(held)), replace=False)
            train_idx = np.concatenate([normal_train, known_train])
            cal_idx = np.concatenate([normal_cal, known_cal])
            test_idx = np.concatenate([normal_test, held_test])
            rng.shuffle(train_idx); rng.shuffle(cal_idx); rng.shuffle(test_idx)
            base = Pipeline([
                ("scaler", MinMaxScaler()),
                ("model", RandomForestClassifier(
                    n_estimators=50, max_depth=20, min_samples_leaf=2,
                    max_features=0.7, class_weight="balanced_subsample",
                    random_state=seed, n_jobs=1,
                )),
            ]).fit(X[train_idx], y[train_idx])
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(base), method=frozen["calibration_method"]
            ).fit(X[cal_idx], y[cal_idx])
            anomaly_scaler = make_scaler(frozen["anomaly_scaler"], seed).fit(X[normal_train])
            detector = train_anomaly(
                frozen["anomaly_model"],
                anomaly_scaler.transform(X[normal_train]).astype(np.float32),
                seed,
            )
            reference = detector.score(
                anomaly_scaler.transform(X[normal_cal]).astype(np.float32)
            )
            classifier = calibrated.predict_proba(X[test_idx])[:, 1]
            anomaly = empirical_probability(
                detector.score(anomaly_scaler.transform(X[test_idx]).astype(np.float32)),
                reference,
            )
            entropy = -(
                np.clip(classifier, 1e-7, 1 - 1e-7) * np.log(np.clip(classifier, 1e-7, 1))
                + np.clip(1 - classifier, 1e-7, 1) * np.log(np.clip(1 - classifier, 1e-7, 1))
            )
            linear = coef[0] * classifier + coef[1] * anomaly + coef[2] * entropy + intercept
            hybrid = 1 / (1 + np.exp(-np.clip(linear, -40, 40)))
            y_test = (family[test_idx] == held_out).astype(np.int8)
            for variant, score, cutoff in (
                ("classifier_only", classifier, 0.5),
                ("frozen_hybrid", hybrid, threshold),
            ):
                rows.append({
                    "seed": seed, "held_out_family": held_out, "variant": variant,
                    **binary_metrics(y_test, score >= cutoff, score),
                })
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"completed unknown seed={seed} family={held_out}", flush=True)
    result = pd.DataFrame(rows)
    summary = result.groupby(["held_out_family", "variant"])[
        ["mcc", "pr_auc", "tpr", "fpr"]
    ].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(
        output["tables"] / "unknown_family_mean_std.csv", index=False
    )
    return result


if __name__ == "__main__":
    run()
