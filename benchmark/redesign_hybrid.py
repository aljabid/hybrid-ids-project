                                                                              

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from benchmark.common import write_json
from benchmark.metrics import binary_metrics
from benchmark.readiness_anomaly import (
    empirical_probability, make_scaler, train_anomaly,
)
from benchmark.redesign_preprocessing import config, load_partition, paths, stratified_sample


ANOMALY_MODELS = [
    "isolation_forest", "autoencoder", "denoising_autoencoder",
    "robust_autoencoder", "variational_autoencoder", "deep_svdd",
]


def _threshold_search(y, probability, fpr_cap=0.02):
    best = None
    for threshold in np.linspace(0.80, 0.999, 80):
        metrics = binary_metrics(y, probability >= threshold, probability)
        if metrics["fpr"] <= fpr_cap and (
            best is None or metrics["mcc"] > best["mcc"]
        ):
            best = {"threshold": float(threshold), **metrics}
    return best or {"threshold": 0.999, **binary_metrics(y, probability >= 0.999, probability)}


def anomaly_selection(seed=42, force=False):
    output = paths()
    target = output["tables"] / "anomaly_selection.csv"
    if target.exists() and not force:
        return pd.read_csv(target)
    X, y, _, _, _ = load_partition("train")
    Xv, yv, _, _, _ = load_partition("validation")
    rng = np.random.default_rng(seed)
    normal = X[y == 0]
    normal_fit = normal[
        rng.choice(len(normal), config()["selection"]["anomaly_normal_rows"], replace=False)
    ]
    val_idx = stratified_sample(yv, config()["selection"]["validation_rows"], seed)
    Xval, yval = Xv[val_idx], yv[val_idx]
    reference_idx, selection_idx = train_test_split(
        np.arange(len(yval)), test_size=0.5, stratify=yval, random_state=seed
    )
    rows = []
    for scaler_name in ("robust", "quantile"):
        scaler = make_scaler(scaler_name, seed).fit(normal_fit)
        normal_scaled = scaler.transform(normal_fit).astype(np.float32)
        val_scaled = scaler.transform(Xval).astype(np.float32)
        for model_name in ANOMALY_MODELS:
            detector = train_anomaly(model_name, normal_scaled, seed, epochs=20)
            reference_score = detector.score(
                val_scaled[reference_idx][yval[reference_idx] == 0]
            )
            probability = empirical_probability(
                detector.score(val_scaled[selection_idx]), reference_score
            )
            selected = _threshold_search(yval[selection_idx], probability)
            rows.append({
                "seed": seed, "scaler": scaler_name, "anomaly_model": model_name,
                **{f"selection_{key}": value for key, value in selected.items()
                   if key in ("threshold", "mcc", "pr_auc", "tpr", "fpr")},
            })
            print(
                f"anomaly {scaler_name}/{model_name}: "
                f"MCC={selected['mcc']:.4f} FPR={selected['fpr']:.4f}",
                flush=True,
            )
    table = pd.DataFrame(rows).sort_values(
        ["selection_mcc", "selection_pr_auc"], ascending=False
    )
    table.to_csv(target, index=False)
    write_json(output["manifests"] / "anomaly_selection.json", {
        "selection_data": "CICIDS2017 validation groups; reference/selection halves",
        "external_labels_used": False, "fpr_cap": 0.02,
        "winner": table.iloc[0].to_dict(),
    })
    return table


def fusion_selection(seed=42, force=False):
    output = paths()
    target = output["tables"] / "conservative_fusion_selection.csv"
    if target.exists() and not force:
        return pd.read_csv(target)
    X, y, _, _, _ = load_partition("train")
    Xv, yv, _, _, _ = load_partition("validation")
    val_idx = stratified_sample(yv, config()["selection"]["validation_rows"], seed)
    Xval, yval = Xv[val_idx], yv[val_idx]
    development_idx, selection_idx = train_test_split(
        np.arange(len(yval)), test_size=0.5, stratify=yval, random_state=seed
    )
    calibration_idx, meta_idx = train_test_split(
        development_idx, test_size=0.5, stratify=yval[development_idx],
        random_state=seed,
    )
    with (output["models"] / "objective_multiclass_family_balanced.pkl").open("rb") as handle:
        classifier = pickle.load(handle)
                                                                      
                                                                        
    raw = classifier.predict_proba(Xval)
    benign_class = int(np.unique(np.load(
        output["data"] / "cicids2017_expanded_train.npz"
    )["family"][y == 0])[0])
    raw_attack = 1 - raw[:, list(classifier.classes_).index(benign_class)]
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_attack[calibration_idx], yval[calibration_idx]
    )
    classifier_probability = calibrator.transform(raw_attack)
    anomaly_choice = json.loads(
        (output["manifests"] / "anomaly_selection.json").read_text()
    )["winner"]
    scaler = make_scaler(anomaly_choice["scaler"], seed)
    rng = np.random.default_rng(seed)
    normal = X[y == 0]
    normal_fit = normal[
        rng.choice(len(normal), config()["selection"]["anomaly_normal_rows"], replace=False)
    ]
    scaler.fit(normal_fit)
    detector = train_anomaly(
        anomaly_choice["anomaly_model"],
        scaler.transform(normal_fit).astype(np.float32), seed, epochs=20,
    )
    scaled = scaler.transform(Xval).astype(np.float32)
    reference = detector.score(
        scaled[calibration_idx][yval[calibration_idx] == 0]
    )
    anomaly_probability = empirical_probability(detector.score(scaled), reference)
    c = classifier_probability[selection_idx]
    a = anomaly_probability[selection_idx]
    ys = yval[selection_idx]
    rows = []
    best_baseline_prediction = None
    best_baseline_mcc = -np.inf
    for classifier_threshold in np.linspace(0.2, 0.8, 13):
        baseline = c >= classifier_threshold
        baseline_metrics = binary_metrics(ys, baseline, c)
        rows.append({
            "strategy": "classifier_only",
            "classifier_threshold": classifier_threshold,
            "anomaly_threshold": np.nan, "uncertainty_floor": np.nan,
            "coverage": 1.0, "corrected_fn": 0, "damaged_tn": 0,
            "net_corrections": 0,
            **baseline_metrics,
        })
        if baseline_metrics["fpr"] <= 0.02 and baseline_metrics["mcc"] > best_baseline_mcc:
            best_baseline_mcc = baseline_metrics["mcc"]
            best_baseline_prediction = baseline.copy()
        for anomaly_threshold in (0.95, 0.975, 0.99, 0.995, 0.999):
            for uncertainty_floor in (0.01, 0.05, 0.10, 0.20):
                rescue = (
                    (~baseline) & (c >= uncertainty_floor)
                    & (c < classifier_threshold) & (a >= anomaly_threshold)
                )
                prediction = baseline | rescue
                score = np.maximum(c, a * rescue)
                metrics = binary_metrics(ys, prediction, score)
                corrected_fn = int((rescue & (ys == 1)).sum())
                damaged_tn = int((rescue & (ys == 0)).sum())
                rows.append({
                    "strategy": "uncertainty_rescue",
                    "classifier_threshold": classifier_threshold,
                    "anomaly_threshold": anomaly_threshold,
                    "uncertainty_floor": uncertainty_floor, "coverage": 1.0,
                    "corrected_fn": corrected_fn, "damaged_tn": damaged_tn,
                    "net_corrections": corrected_fn - damaged_tn, **metrics,
                })
    def meta_features(indices):
        cp = classifier_probability[indices]
        ap = anomaly_probability[indices]
        entropy = -(
            np.clip(cp, 1e-7, 1) * np.log(np.clip(cp, 1e-7, 1))
            + np.clip(1 - cp, 1e-7, 1) * np.log(np.clip(1 - cp, 1e-7, 1))
        )
        return np.column_stack([cp, ap, entropy, np.abs(cp - ap), cp * ap])

    X_meta, X_select = meta_features(meta_idx), meta_features(selection_idx)
    meta_models = {}
    for regularization in (0.1, 1.0, 10.0):
        meta = LogisticRegression(
            C=regularization, class_weight="balanced", random_state=seed,
            max_iter=1000,
        ).fit(X_meta, yval[meta_idx])
        meta_models[regularization] = meta
        stack_probability = meta.predict_proba(X_select)[:, 1]
        for stack_threshold in np.linspace(0.2, 0.9, 29):
            prediction = stack_probability >= stack_threshold
            metrics = binary_metrics(ys, prediction, stack_probability)
            corrected_fn = int(
                ((~best_baseline_prediction) & prediction & (ys == 1)).sum()
            )
            damaged_tn = int(
                ((~best_baseline_prediction) & prediction & (ys == 0)).sum()
            )
            rows.append({
                "strategy": "logistic_meta_gate",
                "classifier_threshold": stack_threshold,
                "anomaly_threshold": np.nan,
                "uncertainty_floor": np.nan, "coverage": 1.0,
                "corrected_fn": corrected_fn, "damaged_tn": damaged_tn,
                "net_corrections": corrected_fn - damaged_tn,
                "meta_C": regularization,
                "meta_coefficients": json.dumps(meta.coef_[0].tolist()),
                "meta_intercept": float(meta.intercept_[0]),
                **metrics,
            })
                                                                         
                                                                          
    for lower, upper in ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)):
        covered = (c <= lower) | (c >= upper)
        pred = c[covered] >= 0.5
        metrics = binary_metrics(ys[covered], pred, c[covered])
        rows.append({
            "strategy": "selective_abstention", "classifier_threshold": 0.5,
            "anomaly_threshold": np.nan, "uncertainty_floor": lower,
            "abstention_upper": upper, "coverage": float(covered.mean()),
            "corrected_fn": 0, "damaged_tn": 0, "net_corrections": 0, **metrics,
        })
    table = pd.DataFrame(rows)
    baseline_fpr = table[table["strategy"] == "classifier_only"].sort_values(
        "mcc", ascending=False
    ).iloc[0]["fpr"]
    eligible = table[
        (table["strategy"].isin([
            "classifier_only", "uncertainty_rescue", "logistic_meta_gate"
        ]))
        & (table["fpr"] <= 0.02)
        & (table["net_corrections"] >= 0)
    ].sort_values(["mcc", "tpr"], ascending=False)
    winner = eligible.iloc[0]
    table.to_csv(target, index=False)
    with (output["models"] / "fusion_components_seed_42.pkl").open("wb") as handle:
        pickle.dump({
            "classifier": classifier, "calibrator": calibrator,
            "anomaly_scaler": scaler, "anomaly_detector": detector,
            "anomaly_reference": reference, "meta_models": meta_models,
        }, handle)
    winner_payload = {
        key: (None if pd.isna(value) else value)
        for key, value in winner.to_dict().items()
    }
    write_json(output["manifests"] / "fusion_selection.json", {
        "selection_data": "CICIDS2017 validation selection half only",
        "external_labels_used": False, "winner": winner_payload,
        "rule": "maximize MCC subject to FPR delta <=0.02 and nonnegative corrections",
    })
    return table


if __name__ == "__main__":
    print(anomaly_selection().head(10).to_string(index=False))
    table = fusion_selection()
    print(table.sort_values("mcc", ascending=False).head(10).to_string(index=False))
