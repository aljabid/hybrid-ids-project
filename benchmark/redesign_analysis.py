                                                                                

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon
from sklearn.metrics import ConfusionMatrixDisplay

from benchmark.common import write_json
from benchmark.metrics import expected_calibration_error
from benchmark.redesign_preprocessing import load_partition, paths


plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 150,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})

METRICS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc",
    "tpr", "fpr", "ece", "brier", "negative_log_likelihood",
]
MODEL_ORDER = [
    "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
    "family_aware_rf", "tabular_resnet", "redesigned_meta_hybrid",
]


def _holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(values) - rank) * values[index])
        previous = max(previous, current)
        adjusted[index] = previous
    return adjusted


def _flatten_summary(frame, groups):
    result = frame.groupby(groups)[METRICS].agg(["mean", "std"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def summaries():
    output = paths()
    runs = pd.read_csv(output["results"] / "grouped_ten_seed_runs.csv")
    summary = _flatten_summary(runs, ["dataset", "model"])
    summary.to_csv(output["tables"] / "benchmark_mean_std.csv", index=False)
    runs.groupby(["dataset", "model"])[METRICS].agg(
        lambda values: f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    ).reset_index().to_csv(
        output["tables"] / "benchmark_formatted.csv", index=False
    )
    calibration = summary[[
        "dataset", "model", "ece_mean", "ece_std", "brier_mean", "brier_std",
        "negative_log_likelihood_mean", "negative_log_likelihood_std",
    ]]
    calibration.to_csv(output["tables"] / "calibration_results.csv", index=False)
    efficiency = runs.groupby(["dataset", "model"])[
        ["training_seconds", "inference_seconds"]
    ].agg(["mean", "std"])
    efficiency.columns = [f"{a}_{b}" for a, b in efficiency.columns]
    efficiency = efficiency.reset_index()
    efficiency["inference_us_per_flow_mean"] = (
        efficiency["inference_seconds_mean"]
        / runs.groupby(["dataset", "model"])["n_samples"].first().values * 1e6
    )
    efficiency.to_csv(output["tables"] / "efficiency_results.csv", index=False)
    return runs, summary


def statistical_tests(runs):
    output = paths()
    rows = []
    comparisons = []
    for dataset in runs["dataset"].unique():
        for baseline in MODEL_ORDER[:-1]:
            comparisons.append((dataset, baseline, "redesigned_meta_hybrid"))
    for dataset, baseline, hybrid in comparisons:
        for metric in ("mcc", "pr_auc", "tpr", "fpr", "ece", "brier"):
            paired = runs[
                (runs["dataset"] == dataset)
                & runs["model"].isin([baseline, hybrid])
            ].pivot(index="seed", columns="model", values=metric).dropna()
            if baseline not in paired or hybrid not in paired:
                continue
            difference = paired[hybrid] - paired[baseline]
            try:
                statistic, pvalue = wilcoxon(difference)
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            rows.append({
                "dataset": dataset, "baseline": baseline, "candidate": hybrid,
                "metric": metric, "n_pairs": len(difference),
                "baseline_mean": paired[baseline].mean(),
                "candidate_mean": paired[hybrid].mean(),
                "candidate_minus_baseline": difference.mean(),
                "wilcoxon_statistic": statistic, "p_raw": pvalue,
            })
    table = pd.DataFrame(rows)
    table["p_holm"] = _holm(table["p_raw"].to_numpy())
    table["significant_0_05"] = table["p_holm"] <= 0.05
    table.to_csv(
        output["tables"] / "statistical_tests_wilcoxon_holm.csv", index=False
    )
    return table


def robustness_statistical_tests():
                                                                                   
    output = paths()
    rows = []
    specifications = [
        (
            output["results"] / "unknown_family_ten_seed_runs.csv",
            ["held_out_family"], "variant", "family_aware_rf",
            "redesigned_meta_hybrid", "unknown_family",
        ),
        (
            output["results"] / "leave_domain_out_selective_runs.csv",
            ["held_out_domain"], "variant", "adaptive_decision_tree",
            "selective_agreement_hybrid", "leave_domain_out",
        ),
        (
            output["results"] / "confirmation_ten_seed_runs.csv",
            [], "variant", "adaptive_decision_tree",
            "selective_agreement_hybrid", "untouched_confirmation",
        ),
    ]
    for source, groups, variant, baseline, candidate, experiment in specifications:
        data = pd.read_csv(source)
        grouped = data.groupby(groups, dropna=False) if groups else [((), data)]
        for key, part in grouped:
            keys = key if isinstance(key, tuple) else (key,)
            for metric in ("mcc", "pr_auc", "tpr", "fpr", "ece", "brier"):
                paired = part[part[variant].isin([baseline, candidate])].pivot(
                    index="seed", columns=variant, values=metric
                ).dropna()
                difference = paired[candidate] - paired[baseline]
                try:
                    statistic, pvalue = wilcoxon(difference)
                except ValueError:
                    statistic, pvalue = 0.0, 1.0
                row = {
                    "experiment": experiment, "metric": metric,
                    "baseline": baseline, "candidate": candidate,
                    "n_pairs": len(difference),
                    "baseline_mean": paired[baseline].mean(),
                    "candidate_mean": paired[candidate].mean(),
                    "candidate_minus_baseline": difference.mean(),
                    "wilcoxon_statistic": statistic, "p_raw": pvalue,
                    "scope_note": (
                        "Selective metrics condition on covered flows; coverage is "
                        "reported separately and direct full-coverage interpretation "
                        "is not valid."
                    ) if "selective" in candidate else "",
                }
                row.update({name: value for name, value in zip(groups, keys)})
                rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = _holm(table["p_raw"].to_numpy())
    table["significant_0_05"] = table["p_holm"] <= .05
    table.to_csv(
        output["tables"] / "robustness_statistical_tests_wilcoxon_holm.csv",
        index=False,
    )
    return table


def bootstrap_intervals(runs, repetitions=20_000):
    rng = np.random.default_rng(42)
    rows = []
    for (dataset, model), part in runs.groupby(["dataset", "model"]):
        for metric in ("mcc", "pr_auc", "tpr", "fpr", "ece", "brier"):
            values = part.sort_values("seed")[metric].to_numpy()
            samples = values[rng.integers(0, len(values), (repetitions, len(values)))]
            means = samples.mean(axis=1)
            rows.append({
                "dataset": dataset, "model": model, "metric": metric,
                "mean": values.mean(), "ci95_low": np.quantile(means, .025),
                "ci95_high": np.quantile(means, .975),
                "bootstrap_repetitions": repetitions,
            })
    table = pd.DataFrame(rows)
    table.to_csv(paths()["tables"] / "bootstrap_confidence_intervals.csv", index=False)
    return table


def dataset_statistics():
    output = paths()
    rows = []
    for partition in ("train", "validation", "internal_test"):
        X, y, family, group, _ = load_partition(partition)
        rows.append({
            "dataset": "CICIDS2017", "partition": partition, "rows": len(y),
            "benign": int((y == 0).sum()), "attack": int((y == 1).sum()),
            "attack_families": int(pd.Series(family[y == 1]).nunique()),
            "collection_groups": int(pd.Series(group).nunique()),
            "features": X.shape[1],
        })
    for name, label in (
        ("cse_cic_ids2018", "CSE-CIC-IDS2018 development diagnostic"),
        ("cic_unsw_nb15", "CIC-UNSW-NB15 development diagnostic"),
    ):
        with np.load(output["data"] / f"{name}_expanded_diagnostic.npz") as data:
            y, family = data["y"], data["family"]
            rows.append({
                "dataset": label, "partition": "whole-domain diagnostic",
                "rows": len(y), "benign": int((y == 0).sum()),
                "attack": int((y == 1).sum()),
                "attack_families": int(pd.Series(family[y == 1]).nunique()),
                "collection_groups": np.nan, "features": data["X"].shape[1],
            })
    confirmation = json.loads(
        (output["manifests"] / "confirmation_dataset.json").read_text()
    )
    metadata = json.loads(
        (output["data"] / "cic_ton_iot_confirmation_aligned/metadata.json").read_text()
    )
    y = np.load(
        output["data"] / "cic_ton_iot_confirmation_aligned/y.npy", mmap_mode="r"
    )[:metadata["valid_rows"]]
    rows.append({
        "dataset": "CIC-ToN-IoT V2 untouched confirmation",
        "partition": "one-shot confirmation", "rows": len(y),
        "benign": int((y == 0).sum()), "attack": int((y == 1).sum()),
        "attack_families": len(metadata["family_to_code"]) - 1,
        "collection_groups": np.nan, "features": 68,
    })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "dataset_statistics.csv", index=False)
    return table


def reliability_and_confusion():
    output = paths()
    prediction_dir = output["results"] / "predictions"
    archives = sorted(prediction_dir.glob("*_seed_42.npz"))
    reliability_rows = []
    for archive in archives:
        if archive.name == "confirmation_seed42.npz":
            continue
        with np.load(archive, allow_pickle=False) as data:
            y, prediction, probability = (
                data["y_true"], data["prediction"], data["probability"]
            )
            _, bins = expected_calibration_error(y, probability)
            model_dataset = archive.stem.removesuffix("_seed_42")
            reliability_rows.extend([
                {"archive": model_dataset, **row} for row in bins
            ])
            fig, axis = plt.subplots(figsize=(2.6, 2.3))
            ConfusionMatrixDisplay.from_predictions(
                y, prediction, labels=[0, 1], display_labels=["Benign", "Attack"],
                cmap="Blues", colorbar=False, ax=axis,
            )
            axis.set_title(model_dataset.replace("_", " "))
            fig.savefig(output["figures"] / f"confusion_{model_dataset}.png")
            plt.close(fig)
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(output["tables"] / "reliability_bins_seed42.csv", index=False)
    for dataset in (
        "cicids2017_group_test", "cse_cic_ids2018", "cic_unsw_nb15"
    ):
        part = reliability[reliability["archive"].str.endswith(dataset)]
        fig, axis = plt.subplots(figsize=(3.5, 2.8))
        axis.plot([.5, 1], [.5, 1], "--", color="0.5", lw=.8)
        for name, values in part.groupby("archive"):
            model = name[:-(len(dataset) + 1)]
            axis.plot(values["confidence"], values["accuracy"], marker="o", ms=2,
                      lw=.8, label=model.replace("_", " "))
        axis.set(xlabel="Predicted-class confidence", ylabel="Empirical accuracy",
                 xlim=(.5, 1), ylim=(.5, 1))
        axis.legend(frameon=False, fontsize=5, ncol=2)
        fig.savefig(output["figures"] / f"reliability_{dataset}.png")
        plt.close(fig)


def confirmation_diagnostics():
    output = paths()
    archive = output["results"] / "predictions" / "confirmation_seed42.npz"
    with np.load(archive, allow_pickle=False) as data:
        y = data["y"]
        dt = data["dt_probability"]
        xgb = data["xgb_probability"]
        covered = data["covered"]
        selective_prediction = data["selective_prediction"]
    seed_row = pd.read_csv(
        output["results"] / "confirmation_ten_seed_runs.csv"
    )
    seed_row = seed_row[seed_row["seed"] == 42].iloc[0]
    adaptive_prediction = dt >= seed_row["dt_threshold"]
    rows = []
    for variant, actual, prediction, probability in (
        ("adaptive_decision_tree", y, adaptive_prediction, dt),
        (
            "selective_agreement_hybrid", y[covered], selective_prediction,
            (dt[covered] + xgb[covered]) / 2,
        ),
    ):
        _, bins = expected_calibration_error(actual, probability)
        rows.extend([{"variant": variant, **item} for item in bins])
        fig, axis = plt.subplots(figsize=(2.6, 2.3))
        ConfusionMatrixDisplay.from_predictions(
            actual, prediction, labels=[0, 1], display_labels=["Benign", "Attack"],
            cmap="Blues", colorbar=False, ax=axis,
        )
        axis.set_title(variant.replace("_", " "))
        fig.savefig(output["figures"] / f"confusion_confirmation_{variant}.png")
        plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(
        output["tables"] / "confirmation_reliability_bins_seed42.csv", index=False
    )
    fig, axis = plt.subplots(figsize=(3.5, 2.8))
    axis.plot([.5, 1], [.5, 1], "--", color=".5", lw=.8)
    for variant, part in table.groupby("variant"):
        axis.plot(part["confidence"], part["accuracy"], marker="o", ms=3,
                  label=variant.replace("_", " "))
    axis.set(xlabel="Predicted-class confidence", ylabel="Empirical accuracy",
             xlim=(.5, 1), ylim=(0, 1))
    axis.legend(frameon=False)
    fig.savefig(output["figures"] / "reliability_cic_ton_iot_confirmation.png")
    plt.close(fig)
    return table


def family_error_analysis():
    output = paths()
    rows = []
    for archive in sorted((output["results"] / "predictions").glob("*_seed_42.npz")):
        if archive.name == "confirmation_seed42.npz":
            continue
        with np.load(archive, allow_pickle=False) as data:
            y, prediction, family = (
                data["y_true"], data["prediction"], data["family"]
            )
        stem = archive.stem.removesuffix("_seed_42")
        for family_id in np.unique(family):
            selected = family == family_id
            rows.append({
                "prediction_archive": stem, "family_id": str(family_id),
                "rows": int(selected.sum()),
                "false_positives": int(
                    ((y[selected] == 0) & (prediction[selected] == 1)).sum()
                ),
                "false_negatives": int(
                    ((y[selected] == 1) & (prediction[selected] == 0)).sum()
                ),
            })
    table = pd.DataFrame(rows)
    table.to_csv(
        output["tables"] / "family_error_analysis_seed42.csv", index=False
    )
    table.sort_values("false_positives", ascending=False).head(25).to_csv(
        output["tables"] / "top_false_positive_families.csv", index=False
    )
    table.sort_values("false_negatives", ascending=False).head(25).to_csv(
        output["tables"] / "top_false_negative_families.csv", index=False
    )
    return table


def hyperparameter_table():
    output = paths()
    rows = []
    for model in ("decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"):
        tuning = json.loads(
            (output["results"] / f"group_tuning_{model}.json").read_text()
        )
        rows.append({
            "model": model,
            "best_parameters": json.dumps(tuning["best_parameters"], sort_keys=True),
            "group_cv_score": tuning.get(
                "best_group_cv_mcc",
                tuning.get("best_cv_mcc", tuning.get("best_score")),
            ),
            "selection_protocol": "GroupKFold on CICIDS2017 training collections",
        })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "model_hyperparameters.csv", index=False)
    return table


def _jsd(base, target):
    combined = np.concatenate([base, target])
    edges = np.unique(np.quantile(combined, np.linspace(0, 1, 51)))
    if len(edges) < 3:
        return 0.0
    p = np.histogram(base, bins=edges)[0] + 1e-9
    q = np.histogram(target, bins=edges)[0] + 1e-9
    return float(jensenshannon(p / p.sum(), q / q.sum()) ** 2)


def drift():
    output = paths()
    rng = np.random.default_rng(42)
    X_train, _, _, _, features = load_partition("train")
    base = X_train[rng.choice(len(X_train), min(100_000, len(X_train)), False)]
    targets = {}
    for name in ("cse_cic_ids2018", "cic_unsw_nb15"):
        with np.load(output["data"] / f"{name}_expanded_diagnostic.npz") as data:
            index = rng.choice(len(data["X"]), min(100_000, len(data["X"])), False)
            targets[name] = data["X"][index]
    metadata = json.loads(
        (output["data"] / "cic_ton_iot_confirmation_aligned/metadata.json").read_text()
    )
    confirmation = np.load(
        output["data"] / "cic_ton_iot_confirmation_aligned/X.npy", mmap_mode="r"
    )[:metadata["valid_rows"]]
    targets["cic_ton_iot_confirmation"] = confirmation[
        rng.choice(len(confirmation), 100_000, False)
    ]
    rows = []
    for dataset, target in targets.items():
        for index, feature in enumerate(features):
            rows.append({
                "dataset": dataset, "feature": feature,
                "js_divergence": _jsd(base[:, index], target[:, index]),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "feature_drift_jsd.csv", index=False)
    matrix = table.pivot(index="feature", columns="dataset", values="js_divergence")
    fig, axis = plt.subplots(figsize=(7.16, 9))
    sns.heatmap(matrix, cmap="magma", vmin=0, vmax=1, ax=axis)
    axis.set_title("Feature drift from CICIDS2017 training (Jensen–Shannon divergence)")
    fig.savefig(output["figures"] / "feature_drift_heatmap.png")
    plt.close(fig)
    return table


def overview_figures(summary):
    output = paths()
                        
    labels = [
        "Real flows\n+ provenance", "68-feature\nsemantic alignment",
        "Group/family\nseparation", "Training-only\nmodel selection",
        "Freeze method\nbefore confirmation", "10-seed statistics\n+ limitations",
    ]
    fig, axis = plt.subplots(figsize=(7.16, 1.8))
    axis.set(xlim=(0, 12), ylim=(0, 2)); axis.axis("off")
    for index, label in enumerate(labels):
        x = .1 + 2 * index
        axis.add_patch(FancyBboxPatch(
            (x, .55), 1.65, .8, boxstyle="round,pad=0.05",
            facecolor="#e8f1fa", edgecolor="#255f85", linewidth=.9,
        ))
        axis.text(x + .825, .95, label, ha="center", va="center", fontsize=7)
        if index < len(labels) - 1:
            axis.annotate("", xy=(x + 1.98, .95), xytext=(x + 1.68, .95),
                          arrowprops={"arrowstyle": "->", "lw": .8})
    fig.savefig(output["figures"] / "overall_benchmark_workflow.png")
    plt.close(fig)
                  
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), sharey=True)
    for axis, dataset in zip(axes, summary["dataset"].unique()):
        part = summary[summary["dataset"] == dataset].set_index("model").reindex(MODEL_ORDER)
        axis.barh(np.arange(len(part)), part["mcc_mean"], xerr=part["mcc_std"], capsize=2)
        axis.axvline(0, color="black", lw=.5)
        axis.set_yticks(np.arange(len(part)), part.index.str.replace("_", " "))
        axis.set_title(dataset.replace("_", " "))
        axis.set_xlabel("MCC (mean ± SD)")
    fig.savefig(output["figures"] / "performance_comparison.png")
    plt.close(fig)
                             
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0), sharey=True)
    for axis, dataset in zip(axes, summary["dataset"].unique()):
        part = summary[summary["dataset"] == dataset].set_index("model").reindex(MODEL_ORDER)
        positions = np.arange(len(part))
        axis.bar(positions - .18, part["ece_mean"], .36, label="ECE")
        axis.bar(positions + .18, part["brier_mean"], .36, label="Brier")
        axis.set_xticks(positions, part.index.str.replace("_", " "),
                        rotation=70, ha="right", fontsize=6)
        axis.set_title(dataset.replace("_", " "))
    axes[0].set_ylabel("Mean calibration error")
    axes[-1].legend(frameon=False)
    fig.savefig(output["figures"] / "calibration_comparison.png")
    plt.close(fig)


def workbook(tables):
    output = paths()
    target = output["tables"] / "manuscript_results.xlsx"
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)


def claims_audit(runs, statistical):
    output = paths()
    unknown = pd.read_csv(
        output["results"] / "unknown_family_ten_seed_runs.csv"
    )
    pivot = unknown.pivot_table(
        index=["seed", "held_out_family"], columns="variant", values="mcc"
    )
    family_means = pivot.groupby("held_out_family").mean()
    hybrid_columns = [name for name in family_means if "hybrid" in name]
    baseline_columns = [name for name in family_means if "classifier" in name or "rf" in name]
    improved = np.nan
    if hybrid_columns and baseline_columns:
        improved = float(
            (family_means[hybrid_columns[0]] > family_means[baseline_columns[0]]).mean()
        )
    audit = {
        "real_data_only": True, "mock_or_synthetic_rows": 0,
        "group_aware_partitioning": True, "temporal_claim_removed": True,
        "confirmation_frozen_before_access": True,
        "confirmation_labels_used_for_revision": False,
        "ten_seed_development_rows": int(len(runs)),
        "unknown_families_tested": int(unknown["held_out_family"].nunique()),
        "unknown_family_mcc_improvement_fraction": improved,
        "consistent_superiority_claim_allowed": False,
        "selective_method_limitations_must_be_reported": True,
        "significance_rows": int(len(statistical)),
    }
    write_json(output["manifests"] / "claims_audit.json", audit)
    return audit


def inventory():
    output = paths()
    rows = []
    for folder in ("tables", "figures", "manifests", "results"):
        for path in sorted(output[folder].rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            rows.append({
                "category": folder,
                "path": str(path.relative_to(output["root"])),
                "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            })
    for path in (
        output["root"] / "README.md",
        output["root"] / "PROTOCOL.md",
        output["root"] / "FULL_PROJECT_REPORT.md",
        output["root"] / "AUTHOR_MANUSCRIPT_HANDOFF_GUIDE.md",
    ):
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "category": "documentation",
            "path": str(path.relative_to(output["root"])),
            "bytes": path.stat().st_size,
            "sha256": digest,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output["root"] / "ARTIFACT_INVENTORY.csv", index=False)
    return table


def main():
    output = paths()
    required = output["manifests"] / "confirmation_execution.json"
    if not required.exists():
        raise RuntimeError("Run the untouched confirmation to completion first")
    runs, summary = summaries()
    statistical = statistical_tests(runs)
    bootstrap = bootstrap_intervals(runs)
    datasets = dataset_statistics()
    reliability_and_confusion()
    drift_table = drift()
    overview_figures(summary)
    audit = claims_audit(runs, statistical)
    workbook({
        "dataset_statistics": datasets, "benchmark_results": summary,
        "calibration_results": pd.read_csv(output["tables"] / "calibration_results.csv"),
        "efficiency_results": pd.read_csv(output["tables"] / "efficiency_results.csv"),
        "statistical_tests": statistical, "bootstrap_ci": bootstrap,
        "unknown_family": pd.read_csv(output["tables"] / "unknown_family_mean_std.csv"),
        "feature_drift": drift_table,
        "confirmation": pd.read_csv(output["tables"] / "confirmation_mean_std.csv"),
    })
    inventory()
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
