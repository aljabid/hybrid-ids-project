                                                                                   

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from benchmark.metrics import binary_metrics
from benchmark.readiness_anomaly import empirical_probability, make_scaler, train_anomaly
from benchmark.redesign_preprocessing import config, load_partition, paths, stratified_sample
from benchmark.redesign_supervised import model_and_space, selected_transformer
from benchmark.redesign_validation import (
    _meta_features, architecture_freeze, split_indices, train_test_indices,
)


def run():
    output = paths()
    checkpoint = output["results"] / "unknown_family_ten_seed_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(int(row["seed"]), int(row["held_out_family"])) for row in rows}
    X, y, family, _, _ = load_partition("train")
    Xv, yv, _, _, _ = load_partition("validation")
    counts = pd.Series(family[y == 1]).value_counts()
    held_families = counts[counts >= 5000].head(5).index.astype(int).tolist()
    architecture = architecture_freeze()
    tuning = json.loads(
        (output["results"] / "group_tuning_random_forest.json").read_text()
    )["best_parameters"]
    for seed in config()["study"]["seeds"]:
        rng = np.random.default_rng(seed)
        normal = rng.permutation(np.flatnonzero(y == 0))
        normal_train, normal_test = normal[:50000], normal[50000:70000]
        validation = stratified_sample(yv, 100000, seed)
        development, selection = train_test_indices(yv[validation], seed)
        calibration_local, meta_local = split_indices(
            development, yv[validation], seed
        )
        calibration, meta = validation[calibration_local], validation[meta_local]
        for held_out in held_families:
            if (seed, held_out) in completed:
                continue
            held = np.flatnonzero(family == held_out)
            known_families = [
                value for value in np.unique(family[y == 1]) if value != held_out
            ]
            known_parts = []
            for value in known_families:
                candidates = np.flatnonzero(family == value)
                known_parts.append(
                    rng.choice(candidates, min(10000, len(candidates)), replace=False)
                )
            known = np.concatenate(known_parts)
            train_index = np.concatenate([normal_train, known])
            rng.shuffle(train_index)
            estimator, _ = model_and_space("random_forest", seed)
            classifier = Pipeline([
                ("preprocessor", selected_transformer(seed)), ("model", estimator)
            ])
            classifier.set_params(**tuning)
            classifier.set_params(model__n_estimators=40)
            classifier.fit(X[train_index], family[train_index])
            benign_class = int(np.unique(family[y == 0])[0])
            classes = list(classifier.classes_)
            val_attack = 1 - classifier.predict_proba(Xv)[:, classes.index(benign_class)]
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                val_attack[calibration], yv[calibration]
            )
            anomaly_scaler = make_scaler(
                architecture["anomaly_scaler"], seed
            ).fit(X[normal_train])
            detector = train_anomaly(
                architecture["anomaly_model"],
                anomaly_scaler.transform(X[normal_train]).astype(np.float32),
                seed, epochs=20,
            )
            val_scaled = anomaly_scaler.transform(Xv).astype(np.float32)
            reference = detector.score(
                val_scaled[calibration][yv[calibration] == 0]
            )
            val_classifier = calibrator.transform(val_attack)
            val_anomaly = empirical_probability(detector.score(val_scaled), reference)
            meta_model = LogisticRegression(
                C=architecture["meta_C"], class_weight="balanced",
                random_state=seed, max_iter=1000,
            ).fit(
                _meta_features(val_classifier[meta], val_anomaly[meta]), yv[meta]
            )
            held_test = rng.choice(held, min(20000, len(held)), replace=False)
            test = np.concatenate([normal_test, held_test])
            rng.shuffle(test)
            y_test = (family[test] == held_out).astype(np.int8)
            raw = classifier.predict_proba(X[test])
            cp = calibrator.transform(1 - raw[:, classes.index(benign_class)])
            ap = empirical_probability(
                detector.score(
                    anomaly_scaler.transform(X[test]).astype(np.float32)
                ),
                reference,
            )
            hp = meta_model.predict_proba(_meta_features(cp, ap))[:, 1]
            for variant, probability, threshold in (
                ("family_aware_rf", cp, 0.80),
                ("redesigned_meta_hybrid", hp, architecture["meta_threshold"]),
            ):
                rows.append({
                    "seed": seed, "held_out_family": held_out,
                    "variant": variant,
                    **binary_metrics(y_test, probability >= threshold, probability),
                })
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"unknown seed={seed} family={held_out}", flush=True)
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
