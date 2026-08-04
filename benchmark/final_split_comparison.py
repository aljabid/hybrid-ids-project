from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from benchmark.common import set_seed
from benchmark.metrics import binary_metrics
from benchmark.redesign_preprocessing import RobustFlowTransformer, config, load_partition, paths


METRICS = ["mcc", "tpr", "fpr", "pr_auc", "ece"]
MODEL_NAMES = ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def model_and_space(name, seed):
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample", random_state=seed, n_jobs=1
        )
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss"
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            random_state=seed, n_jobs=1, class_weight="balanced", verbosity=-1
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            random_seed=seed,
            verbose=False,
            thread_count=1,
            auto_class_weights="Balanced",
            allow_writing_files=False,
        )
    raise KeyError(name)


def selected_transformer(seed=42):
    features = load_partition("train")[4]
    log_indices = tuple(
        features.index(name)
        for name in config()["data"]["log1p_features"]
        if name in features
    )
    return RobustFlowTransformer(
        method="robust", log_indices=log_indices, random_state=seed
    )


def family_balanced_indices(family, y, seed, benign_cap=100000, per_family=20000):
    rng = np.random.default_rng(seed)
    selected = []
    for value in np.unique(family):
        indices = np.flatnonzero(family == value)
        cap = benign_cap if y[indices[0]] == 0 else per_family
        selected.append(rng.choice(indices, min(cap, len(indices)), replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled_cicids2017():
    partitions = [load_partition(name) for name in ("train", "validation", "internal_test")]
    features = partitions[0][4]
    if any(item[4] != features for item in partitions[1:]):
        raise RuntimeError("Feature schema differs across CICIDS2017 partitions")
    X = np.concatenate([item[0] for item in partitions])
    y = np.concatenate([item[1] for item in partitions])
    family = np.concatenate([item[2] for item in partitions])
    source_group = np.concatenate([item[3] for item in partitions])
    return X, y, family, source_group, features


def target_counts():
    result = {}
    for name in ("train", "validation", "internal_test"):
        y = load_partition(name)[1]
        result[name] = {label: int(np.sum(y == label)) for label in (0, 1)}
    return result


def random_row_indices(y: np.ndarray, seed: int, counts: dict):
    rng = np.random.default_rng(seed)
    split = {name: [] for name in ("train", "validation", "internal_test")}
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        cursor = 0
        for name in ("internal_test", "validation", "train"):
            take = counts[name][label]
            split[name].append(indices[cursor : cursor + take])
            cursor += take
        if cursor != len(indices):
            raise RuntimeError(f"Unallocated rows for class {label}: {len(indices) - cursor}")
    for name in split:
        split[name] = np.concatenate(split[name])
        rng.shuffle(split[name])
    return split


def batch_probability(model, X, batch_size=100000):
    return np.concatenate([
        model.predict_proba(X[start : start + batch_size])[:, 1]
        for start in range(0, len(X), batch_size)
    ])


def run_random_row(force=False):
    output = paths()
    target = output["results"] / "matched_random_row_ten_seed_runs.csv"
    rows = [] if force or not target.exists() else pd.read_csv(target).to_dict("records")
    completed = {(int(row["seed"]), row["model"]) for row in rows}
    X, y, family, source_group, features = pooled_cicids2017()
    counts = target_counts()
    seeds = json.loads((output["manifests"] / "final_method_freeze.json").read_text())["seeds"]
    tuning = {
        name: json.loads((output["results"] / f"group_tuning_{name}.json").read_text())
        for name in MODEL_NAMES
    }
    validation_rows = []
    prediction_root = output["results"] / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        split = random_row_indices(y, seed, counts)
        overlap = {
            "train_validation": int(np.intersect1d(split["train"], split["validation"]).size),
            "train_test": int(np.intersect1d(split["train"], split["internal_test"]).size),
            "validation_test": int(np.intersect1d(split["validation"], split["internal_test"]).size),
        }
        for name in split:
            selected_y = y[split[name]]
            validation_rows.append({
                "seed": seed,
                "partition": name,
                "rows": len(split[name]),
                "benign": int(np.sum(selected_y == 0)),
                "attack": int(np.sum(selected_y == 1)),
                "families": int(np.unique(family[split[name]][selected_y == 1]).size),
                "source_groups": int(np.unique(source_group[split[name]]).size),
                **overlap,
            })
        train = split["train"]
        test = split["internal_test"]
        sampled_local = family_balanced_indices(family[train], y[train], seed)
        fit_index = train[sampled_local]
        for name in MODEL_NAMES:
            if (seed, name) in completed:
                continue
            set_seed(seed)
            estimator = model_and_space(name, seed)
            pipeline = Pipeline([
                ("preprocessor", selected_transformer(seed)),
                ("model", estimator),
            ])
            pipeline.set_params(**tuning[name]["best_parameters"])
            started = time.perf_counter()
            pipeline.fit(X[fit_index], y[fit_index])
            training_seconds = time.perf_counter() - started
            started = time.perf_counter()
            probability = batch_probability(pipeline, X[test])
            inference_seconds = time.perf_counter() - started
            prediction = probability >= 0.5
            rows.append({
                "seed": seed,
                "split_strategy": "random_row",
                "model": name,
                "training_rows": len(fit_index),
                "test_rows": len(test),
                "threshold": 0.5,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                **binary_metrics(y[test], prediction, probability),
            })
            pd.DataFrame(rows).to_csv(target, index=False)
            if seed == 42:
                np.savez_compressed(
                    prediction_root / f"matched_random_row_{name}_seed_42.npz",
                    y_true=y[test].astype(np.int8),
                    prediction=prediction.astype(np.int8),
                    probability=probability.astype(np.float32),
                    family=family[test],
                    source_group=source_group[test],
                    pooled_index=test,
                )
            print(f"random-row seed={seed} model={name}", flush=True)
            del pipeline, probability, prediction
            gc.collect()
    validation = pd.DataFrame(validation_rows).drop_duplicates(["seed", "partition"])
    validation.to_csv(output["tables"] / "matched_split_partition_validation.csv", index=False)
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


def analyze():
    output = paths()
    random_runs = pd.read_csv(output["results"] / "matched_random_row_ten_seed_runs.csv")
    group_runs = pd.read_csv(output["results"] / "grouped_ten_seed_runs.csv")
    group_runs = group_runs[
        (group_runs["dataset"] == "cicids2017_group_test")
        & group_runs["model"].isin(MODEL_NAMES)
    ].copy()
    group_runs["split_strategy"] = "group_aware"
    combined = pd.concat([
        group_runs[["seed", "model", "split_strategy", *METRICS]],
        random_runs[["seed", "model", "split_strategy", *METRICS]],
    ], ignore_index=True)
    combined.to_csv(output["tables"] / "matched_split_combined_metrics.csv", index=False)
    summary_rows = []
    test_rows = []
    for model in MODEL_NAMES:
        group = combined[(combined["model"] == model) & (combined["split_strategy"] == "group_aware")].sort_values("seed")
        random = combined[(combined["model"] == model) & (combined["split_strategy"] == "random_row")].sort_values("seed")
        if group["seed"].tolist() != random["seed"].tolist():
            raise RuntimeError(f"Seed pairing failed for {model}")
        summary = {"model": model, "n_pairs": len(group)}
        for metric in METRICS:
            gv = group[metric].to_numpy()
            rv = random[metric].to_numpy()
            delta = rv - gv
            summary.update({
                f"group_{metric}_mean": float(np.mean(gv)),
                f"group_{metric}_std": float(np.std(gv, ddof=1)),
                f"random_{metric}_mean": float(np.mean(rv)),
                f"random_{metric}_std": float(np.std(rv, ddof=1)),
                f"random_minus_group_{metric}_mean": float(np.mean(delta)),
            })
            statistic, p_value = wilcoxon(rv, gv, alternative="two-sided", zero_method="wilcox")
            test_rows.append({
                "model": model,
                "metric": metric,
                "n_pairs": len(gv),
                "group_mean": float(np.mean(gv)),
                "random_mean": float(np.mean(rv)),
                "random_minus_group_mean": float(np.mean(delta)),
                "wilcoxon_statistic": float(statistic),
                "p_raw": float(p_value),
            })
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    tests = holm_adjust(pd.DataFrame(test_rows))
    summary_frame.to_csv(output["tables"] / "matched_split_summary.csv", index=False)
    tests.to_csv(output["tables"] / "matched_split_wilcoxon_holm.csv", index=False)
    make_figure(summary_frame, output["figures"])
    validation = pd.read_csv(output["tables"] / "matched_split_partition_validation.csv")
    manifest = {
        "status": "VALIDATED" if (
            len(random_runs) == 50
            and len(group_runs) == 50
            and validation[["train_validation", "train_test", "validation_test"]].to_numpy().max() == 0
        ) else "INVALID",
        "research_question_role": "Matched random-row versus group-aware comparison",
        "population": "Same de-duplicated CICIDS2017 1,959,228-flow pool",
        "partition_matching": "Exact group-aware row and binary-class counts for train, validation, and test",
        "only_intended_difference": "Row assignment strategy",
        "preprocessing": "Identical RobustFlowTransformer specification; fit on each training split only",
        "feature_schema": "Identical 68-feature semantic schema",
        "hyperparameters": "Identical frozen group-tuned parameters for each model",
        "models": MODEL_NAMES,
        "seeds": sorted(random_runs["seed"].unique().astype(int).tolist()),
        "metrics": METRICS,
        "random_rows": len(random_runs),
        "group_rows": len(group_runs),
        "partition_overlap_max": int(validation[["train_validation", "train_test", "validation_test"]].to_numpy().max()),
        "family_composition_caveat": "Binary counts and sizes are matched; attack-family composition differs as a consequence of split strategy",
        "input_sha256": {
            name: file_sha256(output["data"] / f"cicids2017_expanded_{name}.npz")
            for name in ("train", "validation", "internal_test")
        },
        "result_sha256": file_sha256(output["results"] / "matched_random_row_ten_seed_runs.csv"),
    }
    (output["manifests"] / "matched_split_validation.json").write_text(json.dumps(manifest, indent=2))
    if manifest["status"] != "VALIDATED":
        raise RuntimeError(json.dumps(manifest, indent=2))
    return summary_frame, tests, manifest


def make_figure(summary: pd.DataFrame, figure_root: Path):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    x = np.arange(len(summary))
    width = 0.35
    fig, axis = plt.subplots(figsize=(7.16, 2.55), constrained_layout=True)
    axis.bar(
        x - width / 2,
        summary["group_mcc_mean"],
        width,
        yerr=summary["group_mcc_std"],
        label="Group-aware",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
    )
    axis.bar(
        x + width / 2,
        summary["random_mcc_mean"],
        width,
        yerr=summary["random_mcc_std"],
        label="Random-row",
        color="#E69F00",
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        capsize=2,
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_ylabel("MCC (mean ± SD)")
    axis.set_xticks(x)
    axis.set_xticklabels([MODEL_LABELS[name] for name in summary["model"]])
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    axis.grid(axis="y", color="0.85", linewidth=0.5)
    for suffix in ("pdf", "svg"):
        fig.savefig(figure_root / f"matched_split_mcc.{suffix}", bbox_inches="tight")
    fig.savefig(figure_root / "matched_split_mcc.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    run_random_row()
    summary, tests, manifest = analyze()
    print(summary.to_string(index=False), flush=True)
    print(tests.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
