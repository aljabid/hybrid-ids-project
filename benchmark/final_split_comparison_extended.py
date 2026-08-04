"""Extend the matched random-row-versus-group-aware control (Table II) from
the five mandatory tree/boosting models to all eight benchmarked
configurations: family-aware multiclass Random Forest, the redesigned
meta-fusion hybrid, and the tabular residual network.

This mirrors benchmark.redesign_validation.run_classical/run_deep exactly,
except every array comes from the random-row partitions built by
benchmark.final_split_comparison instead of the group-aware partitions on
disk. Frozen choices (anomaly scaler/model, meta-gate C, meta threshold,
random-forest hyperparameters) are reused unmodified from
manifests/architecture_freeze.json and results/group_tuning_random_forest.json
so the only varied factor remains the split strategy, matching the
final_split_comparison.py contract.
"""

from __future__ import annotations

import gc
import json
import time

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark.common import set_seed
from benchmark.metrics import binary_metrics
from benchmark.readiness_anomaly import empirical_probability, make_scaler, train_anomaly
from benchmark.redesign_preprocessing import config, paths, stratified_sample
from benchmark.redesign_supervised import (
    TabularResNet, family_balanced_indices, model_and_space, selected_transformer,
)
from benchmark.final_split_comparison import (
    MODEL_NAMES as MANDATORY_MODELS,
    METRICS,
    pooled_cicids2017,
    random_row_indices,
    target_counts,
)

EXTENDED_MODELS = ["family_aware_rf", "redesigned_meta_hybrid", "tabular_resnet"]
ALL_MODELS = MANDATORY_MODELS + EXTENDED_MODELS
LABELS = {
    "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "family_aware_rf": "Family-aware RF",
    "redesigned_meta_hybrid": "Meta hybrid",
    "tabular_resnet": "Tabular ResNet",
}


def _meta_features(classifier_probability, anomaly_probability):
    cp, ap = classifier_probability, anomaly_probability
    entropy = -(
        np.clip(cp, 1e-7, 1) * np.log(np.clip(cp, 1e-7, 1))
        + np.clip(1 - cp, 1e-7, 1) * np.log(np.clip(1 - cp, 1e-7, 1))
    )
    return np.column_stack([cp, ap, entropy, np.abs(cp - ap), cp * ap])


def _probability(model, X, batch=100000):
    return np.concatenate([
        model.predict_proba(X[start:start + batch]) for start in range(0, len(X), batch)
    ])


def _dev_split(y, seed):
    development, selection = train_test_split(
        np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed
    )
    calibration, meta = train_test_split(
        development, test_size=0.5, stratify=y[development], random_state=seed
    )
    return calibration, meta


def _record(rows, seed, model, y, family, probability, threshold,
            training_seconds, inference_seconds, prediction_root):
    prediction = probability >= threshold
    rows.append({
        "seed": seed, "split_strategy": "random_row", "model": model,
        "threshold": threshold, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        **binary_metrics(y, prediction, probability),
    })
    if seed == 42:
        np.savez_compressed(
            prediction_root / f"matched_random_row_{model}_seed_42.npz",
            y_true=y.astype(np.int8), prediction=prediction.astype(np.int8),
            probability=probability.astype(np.float32), family=family,
        )


def _train_deep_on(X, y, family, Xval, yval, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    train_index = family_balanced_indices(family, y, seed, benign_cap=150000, per_family=30000)
    cap = config()["selection"]["deep_training_rows"]
    if len(train_index) > cap:
        train_index = rng.choice(train_index, cap, replace=False)
    preprocessor = selected_transformer(seed).fit(X[train_index])
    Xt = preprocessor.transform(X[train_index])
    Xv = preprocessor.transform(Xval)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(Xt), torch.as_tensor(y[train_index], dtype=torch.float32)),
        batch_size=2048, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    model = TabularResNet(X.shape[1])
    positive_weight = torch.tensor(
        [(y[train_index] == 0).sum() / max((y[train_index] == 1).sum(), 1)],
        dtype=torch.float32,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    Xval_tensor = torch.as_tensor(Xv)
    best_mcc, best_state, stale = -1.0, None, 0
    for epoch in range(config()["selection"]["deep_epochs"]):
        model.train()
        for batch, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(model(Xval_tensor)).numpy()
        metrics = binary_metrics(yval, probability >= 0.5, probability)
        scheduler.step(metrics["mcc"])
        if metrics["mcc"] > best_mcc + 1e-4:
            best_mcc, stale = metrics["mcc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"[random-row resnet] seed={seed} epoch={epoch + 1} val_mcc={metrics['mcc']:.4f}", flush=True)
        if stale >= config()["selection"]["deep_patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, preprocessor


def run_extended(force=False):
    output = paths()
    target = output["results"] / "matched_random_row_extended_ten_seed_runs.csv"
    rows = [] if force or not target.exists() else pd.read_csv(target).to_dict("records")
    completed = {(int(row["seed"]), row["model"]) for row in rows}
    architecture = json.loads((output["manifests"] / "architecture_freeze.json").read_text())
    rf_tuning = json.loads((output["results"] / "group_tuning_random_forest.json").read_text())
    X, y, family, source_group, features = pooled_cicids2017()
    counts = target_counts()
    seeds = json.loads((output["manifests"] / "final_method_freeze.json").read_text())["seeds"]
    prediction_root = output["results"] / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        split = random_row_indices(y, seed, counts)
        train_idx, val_idx, test_idx = split["train"], split["validation"], split["internal_test"]
        Xtr, ytr, family_tr = X[train_idx], y[train_idx], family[train_idx]
        Xval_pool, yval_pool = X[val_idx], y[val_idx]
        Xtest, ytest, family_test = X[test_idx], y[test_idx], family[test_idx]
        val_sample = stratified_sample(yval_pool, config()["selection"]["validation_rows"], seed)
        Xval, yval = Xval_pool[val_sample], yval_pool[val_sample]

        need_family = (seed, "family_aware_rf") not in completed
        need_hybrid = (seed, "redesigned_meta_hybrid") not in completed
        if need_family or need_hybrid:
            fit_index = family_balanced_indices(family_tr, ytr, seed)
            estimator, _ = model_and_space("random_forest", seed)
            from sklearn.pipeline import Pipeline
            classifier = Pipeline([
                ("preprocessor", selected_transformer(seed)), ("model", estimator),
            ])
            classifier.set_params(**rf_tuning["best_parameters"])
            started = time.perf_counter()
            classifier.fit(Xtr[fit_index], family_tr[fit_index])
            classifier_training = time.perf_counter() - started

            benign_class = int(np.unique(family_tr[ytr == 0])[0])
            classes = list(classifier.classes_)
            val_raw = _probability(classifier, Xval)
            val_attack = 1 - val_raw[:, classes.index(benign_class)]
            calibration, meta = _dev_split(yval, seed)
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                val_attack[calibration], yval[calibration]
            )

            rng = np.random.default_rng(seed)
            normal = Xtr[ytr == 0]
            normal_fit = normal[
                rng.choice(len(normal), config()["selection"]["anomaly_normal_rows"], replace=False)
            ]
            anomaly_scaler = make_scaler(architecture["anomaly_scaler"], seed).fit(normal_fit)
            anomaly_detector = train_anomaly(
                architecture["anomaly_model"],
                anomaly_scaler.transform(normal_fit).astype(np.float32), seed, epochs=20,
            )
            val_scaled = anomaly_scaler.transform(Xval).astype(np.float32)
            reference = anomaly_detector.score(val_scaled[calibration][yval[calibration] == 0])
            val_anomaly = empirical_probability(anomaly_detector.score(val_scaled), reference)
            val_calibrated = calibrator.transform(val_attack)
            meta_model = LogisticRegression(
                C=architecture["meta_C"], class_weight="balanced", random_state=seed, max_iter=1000,
            ).fit(_meta_features(val_calibrated[meta], val_anomaly[meta]), yval[meta])
            hybrid_training = time.perf_counter() - started

            started_inference = time.perf_counter()
            raw = _probability(classifier, Xtest)
            classifier_probability = calibrator.transform(1 - raw[:, classes.index(benign_class)])
            anomaly_probability = empirical_probability(
                anomaly_detector.score(anomaly_scaler.transform(Xtest).astype(np.float32)), reference,
            )
            meta_probability = meta_model.predict_proba(
                _meta_features(classifier_probability, anomaly_probability)
            )[:, 1]
            inference = time.perf_counter() - started_inference

            if need_family:
                _record(rows, seed, "family_aware_rf", ytest, family_test,
                        classifier_probability, 0.80, classifier_training, inference, prediction_root)
            if need_hybrid:
                _record(rows, seed, "redesigned_meta_hybrid", ytest, family_test,
                        meta_probability, architecture["meta_threshold"], hybrid_training,
                        inference, prediction_root)
            pd.DataFrame(rows).to_csv(target, index=False)
            print(f"random-row extended seed={seed} family_aware_rf+redesigned_meta_hybrid done", flush=True)
            del classifier, anomaly_detector
            gc.collect()

        if (seed, "tabular_resnet") not in completed:
            started = time.perf_counter()
            model, preprocessor = _train_deep_on(Xtr, ytr, family_tr, Xval, yval, seed)
            training = time.perf_counter() - started
            started_inference = time.perf_counter()
            probabilities = []
            with torch.no_grad():
                for start in range(0, len(Xtest), 50000):
                    transformed = preprocessor.transform(Xtest[start:start + 50000])
                    probabilities.append(torch.sigmoid(model(torch.as_tensor(transformed))).numpy())
            probability = np.concatenate(probabilities)
            inference = time.perf_counter() - started_inference
            _record(rows, seed, "tabular_resnet", ytest, family_test, probability, 0.5,
                    training, inference, prediction_root)
            pd.DataFrame(rows).to_csv(target, index=False)
            print(f"random-row extended seed={seed} tabular_resnet done", flush=True)
            del model, preprocessor
            gc.collect()

    return pd.DataFrame(rows)


def holm_adjust(frame: pd.DataFrame):
    order = np.argsort(frame["p_raw"].to_numpy())
    adjusted = np.empty(len(frame), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, frame.iloc[index]["p_raw"] * (len(frame) - rank))
        running = max(running, value)
        adjusted[index] = running
    result = frame.copy()
    result["p_holm"] = adjusted
    result["significant_0_05"] = result["p_holm"] < 0.05
    return result


def analyze_all():
    output = paths()
    mandatory_random = pd.read_csv(output["results"] / "matched_random_row_ten_seed_runs.csv")
    extended_random = pd.read_csv(output["results"] / "matched_random_row_extended_ten_seed_runs.csv")
    random_runs = pd.concat([mandatory_random, extended_random], ignore_index=True)

    group_runs = pd.read_csv(output["results"] / "grouped_ten_seed_runs.csv")
    group_runs = group_runs[
        (group_runs["dataset"] == "cicids2017_group_test") & group_runs["model"].isin(ALL_MODELS)
    ].copy()
    group_runs["split_strategy"] = "group_aware"

    combined = pd.concat([
        group_runs[["seed", "model", "split_strategy", *METRICS]],
        random_runs[["seed", "model", "split_strategy", *METRICS]],
    ], ignore_index=True)
    combined.to_csv(output["tables"] / "matched_split_combined_metrics.csv", index=False)

    summary_rows, test_rows = [], []
    for model in ALL_MODELS:
        group = combined[(combined["model"] == model) & (combined["split_strategy"] == "group_aware")].sort_values("seed")
        random = combined[(combined["model"] == model) & (combined["split_strategy"] == "random_row")].sort_values("seed")
        if group["seed"].tolist() != random["seed"].tolist():
            raise RuntimeError(f"Seed pairing failed for {model}")
        summary = {"model": model, "n_pairs": len(group)}
        for metric in METRICS:
            gv, rv = group[metric].to_numpy(), random[metric].to_numpy()
            delta = rv - gv
            summary.update({
                f"group_{metric}_mean": float(np.mean(gv)), f"group_{metric}_std": float(np.std(gv, ddof=1)),
                f"random_{metric}_mean": float(np.mean(rv)), f"random_{metric}_std": float(np.std(rv, ddof=1)),
                f"random_minus_group_{metric}_mean": float(np.mean(delta)),
            })
            statistic, p_value = wilcoxon(rv, gv, alternative="two-sided", zero_method="wilcox")
            test_rows.append({
                "model": model, "metric": metric, "n_pairs": len(gv),
                "group_mean": float(np.mean(gv)), "random_mean": float(np.mean(rv)),
                "random_minus_group_mean": float(np.mean(delta)),
                "wilcoxon_statistic": float(statistic), "p_raw": float(p_value),
            })
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    tests = holm_adjust(pd.DataFrame(test_rows))
    summary_frame.to_csv(output["tables"] / "matched_split_summary.csv", index=False)
    tests.to_csv(output["tables"] / "matched_split_wilcoxon_holm.csv", index=False)
    make_figure(summary_frame, output["figures"])

    manifest = {
        "status": "VALIDATED" if (
            len(random_runs) == len(ALL_MODELS) * 10 and len(group_runs) == len(ALL_MODELS) * 10
        ) else "INVALID",
        "research_question_role": "Matched random-row versus group-aware comparison, extended to all eight benchmarked configurations",
        "models": ALL_MODELS,
        "extended_models": EXTENDED_MODELS,
        "extension_note": (
            "family_aware_rf, redesigned_meta_hybrid, and tabular_resnet were added after the "
            "initial five-mandatory-model control. Frozen architecture choices, random-forest "
            "hyperparameters, anomaly scaler/model, meta-gate C, and decision thresholds are "
            "reused unmodified from architecture_freeze.json / group_tuning_random_forest.json; "
            "only the split strategy varies, matching the original control's contract."
        ),
        "random_rows": len(random_runs), "group_rows": len(group_runs),
    }
    (output["manifests"] / "matched_split_validation_extended.json").write_text(json.dumps(manifest, indent=2))
    if manifest["status"] != "VALIDATED":
        raise RuntimeError(json.dumps(manifest, indent=2))
    return summary_frame, tests, manifest


def make_figure(summary: pd.DataFrame, figure_root):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    order = [m for m in ALL_MODELS if m in summary["model"].tolist()]
    summary = summary.set_index("model").loc[order].reset_index()
    x = np.arange(len(summary))
    width = 0.35
    fig, axis = plt.subplots(figsize=(7.16, 2.75), constrained_layout=True)
    axis.bar(
        x - width / 2, summary["group_mcc_mean"], width, yerr=summary["group_mcc_std"],
        label="Group-aware", color="#0072B2", edgecolor="black", linewidth=0.5, capsize=2,
    )
    axis.bar(
        x + width / 2, summary["random_mcc_mean"], width, yerr=summary["random_mcc_std"],
        label="Random-row", color="#E69F00", edgecolor="black", linewidth=0.5, hatch="///", capsize=2,
    )
    axis.axvline(len(MANDATORY_MODELS) - 0.5, color="0.6", linewidth=0.6, linestyle=":")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_ylabel("MCC (mean +/- SD)")
    axis.set_xticks(x)
    axis.set_xticklabels([LABELS[name] for name in summary["model"]], rotation=20, ha="right")
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    axis.grid(axis="y", color="0.85", linewidth=0.5)
    for suffix in ("pdf", "svg"):
        fig.savefig(figure_root / f"matched_split_mcc.{suffix}", bbox_inches="tight")
    fig.savefig(figure_root / "matched_split_mcc.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    run_extended()
    summary, tests, manifest = analyze_all()
    print(summary.to_string(index=False), flush=True)
    print(tests.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
