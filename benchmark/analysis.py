from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import confusion_matrix
from benchmark.common import load_config, resolve, write_json
from benchmark.metrics import expected_calibration_error
from benchmark.hybrid_experiments import fixed_router
from benchmark.model_registry import MODEL_NAMES
from benchmark.supervised import load_partition
MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
}
PALETTE = {
    "decision_tree": "#4C78A8",
    "random_forest": "#59A14F",
    "xgboost": "#F28E2B",
    "lightgbm": "#E15759",
    "catboost": "#B07AA1",
    "mlp": "#76B7B2",
}
def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
def _save(fig: plt.Figure, name: str) -> None:
    out = resolve(f"artifacts/figures/{name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)
def statistical_tests() -> pd.DataFrame:
    supervised = pd.read_csv(resolve("artifacts/results/supervised_seed_results.csv"))
    hybrid_path = resolve("artifacts/results/hybrid_seed_results.csv")
    rows = []
    comparisons = []
    if hybrid_path.exists():
        hybrid = pd.read_csv(hybrid_path)
        backbone = hybrid["backbone"].iloc[0]
        for dataset in hybrid["dataset"].unique():
            for metric in ("mcc", "f1", "pr_auc", "ece"):
                baseline = supervised[
                    (supervised.model == backbone) & (supervised.dataset == dataset)
                ].sort_values("seed")
                for variant in ("fixed_hybrid", "confidence_hybrid", "adaptive_hybrid"):
                    proposed = hybrid[
                        (hybrid.variant == variant) & (hybrid.dataset == dataset)
                    ].sort_values("seed")
                    comparisons.append(
                        (dataset, metric, backbone, variant, baseline, proposed)
                    )
    internal = supervised[supervised.dataset == "cicids2017_internal"]
    best = internal.groupby("model").mcc.mean().idxmax()
    for dataset in supervised.dataset.unique():
        for metric in ("mcc", "f1", "pr_auc", "roc_auc"):
            reference = supervised[
                (supervised.model == best) & (supervised.dataset == dataset)
            ].sort_values("seed")
            for challenger in MODEL_NAMES:
                if challenger == best:
                    continue
                compared = supervised[
                    (supervised.model == challenger) & (supervised.dataset == dataset)
                ].sort_values("seed")
                comparisons.append(
                    (dataset, metric, best, challenger, reference, compared)
                )
    unknown_path = resolve("artifacts/results/unknown_attack_seed_results.csv")
    if unknown_path.exists():
        unknown = pd.read_csv(unknown_path)
        for family in unknown.held_out_attack_family.unique():
            for metric in ("mcc", "f1", "recall", "pr_auc", "fpr"):
                reference = unknown[
                    (unknown.held_out_attack_family == family)
                    & (unknown.variant == "classifier_only")
                ].sort_values("seed")
                compared = unknown[
                    (unknown.held_out_attack_family == family)
                    & (unknown.variant == "unknown_attack_hybrid")
                ].sort_values("seed")
                comparisons.append(
                    (
                        f"unknown:{family}",
                        metric,
                        "classifier_only",
                        "unknown_attack_hybrid",
                        reference,
                        compared,
                    )
                )
    for dataset, metric, reference_name, compared_name, reference, compared in comparisons:
        merged = reference[["seed", metric]].merge(
            compared[["seed", metric]], on="seed", suffixes=("_reference", "_compared")
        )
        if len(merged) < 2:
            continue
        differences = merged[f"{metric}_compared"] - merged[f"{metric}_reference"]
        try:
            statistic, p_value = wilcoxon(differences, zero_method="zsplit")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        nonzero = differences[differences != 0]
        rank_biserial = (
            float((nonzero > 0).sum() - (nonzero < 0).sum()) / len(nonzero)
            if len(nonzero)
            else 0.0
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "reference": reference_name,
                "compared": compared_name,
                "n_pairs": len(merged),
                "median_difference_compared_minus_reference": float(np.median(differences)),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
                "rank_biserial_effect": rank_biserial,
                "wins": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "losses": int((differences < 0).sum()),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        order = np.argsort(table.p_value.to_numpy())
        adjusted = np.empty(len(table))
        running = 0.0
        for rank, index in enumerate(order):
            value = min(1.0, table.p_value.iloc[index] * (len(table) - rank))
            running = max(running, value)
            adjusted[index] = running
        table["holm_adjusted_p"] = adjusted
        table["significant_0_05"] = table.holm_adjusted_p < 0.05
    table.to_csv(resolve("artifacts/tables/statistical_tests.csv"), index=False)
    return table
def calibration_figures() -> pd.DataFrame:
    rows = []
    for dataset in ("cicids2017_internal", "cic_ids2018_external"):
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.5), sharex=True, sharey=True)
        for axis, model in zip(axes.ravel(), MODEL_NAMES):
            path = resolve(
                f"artifacts/results/predictions/{model}_{dataset}_seed_42.npz"
            )
            if not path.exists():
                axis.set_visible(False)
                continue
            data = np.load(path)
            ece, bins = expected_calibration_error(data["y_true"], data["probability"])
            for item in bins:
                rows.append({"model": model, "dataset": dataset, "ece": ece, **item})
            axis.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
            axis.plot(
                [item["confidence"] for item in bins],
                [item["accuracy"] for item in bins],
                marker="o",
                color=PALETTE[model],
                linewidth=1.2,
            )
            axis.set_title(f"{MODEL_LABELS[model]} (ECE={ece:.3f})")
            axis.set_xlabel("Confidence")
            axis.set_ylabel("Accuracy")
        fig.suptitle(f"Reliability Diagrams — {dataset.replace('_', ' ').title()}")
        fig.tight_layout()
        _save(fig, f"calibration_{dataset}")
    table = pd.DataFrame(rows)
    table.to_csv(resolve("artifacts/results/calibration_bins.csv"), index=False)
    return table
def performance_figure() -> None:
    results = pd.read_csv(resolve("artifacts/results/supervised_seed_results.csv"))
    plot = (
        results.groupby(["model", "dataset"])[["mcc", "f1", "pr_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharey=False)
    for axis, metric in zip(axes, ("mcc", "f1", "pr_auc")):
        for index, dataset in enumerate(plot.dataset.unique()):
            subset = plot[plot.dataset == dataset]
            x = np.arange(len(subset)) + (index - 0.5) * 0.18
            axis.bar(
                x,
                subset[(metric, "mean")],
                yerr=subset[(metric, "std")].fillna(0),
                width=0.18,
                label=dataset.replace("_", " "),
                capsize=2,
            )
        axis.set_xticks(np.arange(len(MODEL_NAMES)))
        axis.set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES], rotation=50, ha="right")
        axis.set_title(metric.upper().replace("_", "-"))
        axis.set_ylim(bottom=min(0, axis.get_ylim()[0]))
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove() if axes[0].legend_ else None
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "supervised_performance_comparison")
def summary_tables() -> None:
    results = pd.read_csv(resolve("artifacts/results/supervised_seed_results.csv"))
    calibration = (
        results.groupby(["model", "dataset"])[
            ["ece", "brier", "negative_log_likelihood"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    calibration.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in calibration.columns
    ]
    calibration.to_csv(resolve("artifacts/tables/calibration_summary.csv"), index=False)
    efficiency = (
        results.groupby(["model", "dataset"])[
            [
                "training_seconds", "inference_seconds", "microseconds_per_flow",
                "flows_per_second", "peak_rss_mb", "model_size_mb",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    efficiency.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in efficiency.columns
    ]
    efficiency.to_csv(resolve("artifacts/tables/efficiency_summary.csv"), index=False)
def hybrid_ablation_figure() -> None:
    path = resolve("artifacts/results/hybrid_seed_results.csv")
    if not path.exists():
        return
    results = pd.read_csv(path)
    order = [
        "classifier_only", "autoencoder_only", "fixed_hybrid",
        "confidence_hybrid", "adaptive_hybrid",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    for axis, metric in zip(axes, ("mcc", "tpr", "fpr")):
        summary = (
            results.groupby(["variant", "dataset"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        for index, dataset in enumerate(summary.dataset.unique()):
            subset = summary[summary.dataset == dataset].set_index("variant").reindex(order)
            x = np.arange(len(order)) + (index - 0.5) * 0.18
            axis.bar(
                x, subset["mean"], yerr=subset["std"].fillna(0), width=0.18,
                capsize=2, label=dataset.replace("_", " "),
            )
        axis.set_xticks(np.arange(len(order)))
        axis.set_xticklabels(
            ["Classifier", "AE", "Fixed", "Confidence", "Adaptive"],
            rotation=45, ha="right",
        )
        axis.set_title(metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove() if axes[0].legend_ else None
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "hybrid_ablation_comparison")
def routing_sensitivity() -> pd.DataFrame:
    rows = []
    for dataset in ("cicids2017_internal", "cic_ids2018_external"):
        path = resolve(
            f"artifacts/results/hybrid_predictions/adaptive_hybrid_{dataset}_seed_42.npz"
        )
        if not path.exists():
            continue
        data = np.load(path)
        for confidence_threshold in np.linspace(0.50, 0.99, 20):
            prediction, score, route = fixed_router(
                data["attack_probability"],
                data["anomaly_probability"],
                float(confidence_threshold),
                0.99,
            )
            from benchmark.metrics import binary_metrics
            metrics = binary_metrics(data["y_true"], prediction, score)
            rows.append(
                {
                    "dataset": dataset,
                    "confidence_threshold": confidence_threshold,
                    "classifier_coverage": float(route.mean()),
                    "selective_risk": float(1 - metrics["accuracy"]),
                    "mcc": metrics["mcc"],
                    "tpr": metrics["tpr"],
                    "fpr": metrics["fpr"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(resolve("artifacts/tables/routing_threshold_sensitivity.csv"), index=False)
    if not table.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
        for dataset, subset in table.groupby("dataset"):
            axes[0].plot(
                subset.classifier_coverage, subset.selective_risk,
                marker="o", markersize=2.5, label=dataset.replace("_", " "),
            )
            axes[1].plot(
                subset.confidence_threshold, subset.mcc,
                marker="o", markersize=2.5, label=dataset.replace("_", " "),
            )
        axes[0].set_xlabel("Classifier coverage")
        axes[0].set_ylabel("Risk (1 − accuracy)")
        axes[0].set_title("Risk–Coverage")
        axes[1].set_xlabel("Confidence threshold")
        axes[1].set_ylabel("MCC")
        axes[1].set_title("Threshold Sensitivity")
        axes[0].legend(frameon=False)
        fig.tight_layout()
        _save(fig, "routing_risk_coverage")
    return table
def drift_analysis() -> pd.DataFrame:
    X_train, _, features = load_partition("cicids2017_train")
    X_external, _, _ = load_partition("cic_ids2018_external")
    rows = []
    for index, feature in enumerate(features):
        train = X_train[:, index].astype(float)
        external = X_external[:, index].astype(float)
        edges = np.unique(np.quantile(train, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            minimum = min(train.min(), external.min())
            maximum = max(train.max(), external.max())
            edges = np.linspace(minimum, maximum + 1e-12, 11)
        train_hist, _ = np.histogram(train, bins=edges)
        external_hist, _ = np.histogram(external, bins=edges)
        train_dist = (train_hist + 1e-6) / (train_hist.sum() + 1e-6 * len(train_hist))
        external_dist = (external_hist + 1e-6) / (
            external_hist.sum() + 1e-6 * len(external_hist)
        )
        js = float(jensenshannon(train_dist, external_dist, base=2) ** 2)
        psi = float(np.sum((external_dist - train_dist) * np.log(external_dist / train_dist)))
        outside = float(((external < train.min()) | (external > train.max())).mean())
        rows.append(
            {
                "feature": feature,
                "jensen_shannon_divergence": js,
                "population_stability_index": psi,
                "external_outside_training_range": outside,
            }
        )
    table = pd.DataFrame(rows).sort_values("jensen_shannon_divergence", ascending=False)
    table.to_csv(resolve("artifacts/tables/feature_drift.csv"), index=False)
    heatmap_data = table.set_index("feature")[
        ["jensen_shannon_divergence", "population_stability_index", "external_outside_training_range"]
    ]
    normalized = (heatmap_data - heatmap_data.min()) / (
        heatmap_data.max() - heatmap_data.min() + 1e-12
    )
    fig, axis = plt.subplots(figsize=(3.5, 3.7))
    sns.heatmap(normalized, annot=heatmap_data, fmt=".3f", cmap="viridis", ax=axis)
    axis.set_title("CICIDS2017 → CSE-CIC-IDS2018 Feature Drift")
    axis.set_xlabel("Drift metric (color normalized)")
    fig.tight_layout()
    _save(fig, "feature_drift_heatmap")
    return table
def shap_analysis(sample_size: int = 2000) -> pd.DataFrame:
    import shap
    X_internal, y_internal, features = load_partition("cicids2017_test")
    X_external, y_external, _ = load_partition("cic_ids2018_external")
    rng = np.random.default_rng(42)
    frames = []
    for model_name in MODEL_NAMES:
        path = resolve(f"artifacts/models/{model_name}_seed_42.pkl")
        if not path.exists():
            continue
        with path.open("rb") as handle:
            pipeline = pickle.load(handle)
        estimator = pipeline.named_steps["model"]
        transform = pipeline[:-1]
        selected = np.asarray(features)[
            np.arange(len(features))
            if pipeline.named_steps["selector"] == "passthrough"
            else pipeline.named_steps["selector"].get_support(indices=True)
        ]
        domain_importance = {}
        for domain, X, y in (
            ("cicids2017_internal", X_internal, y_internal),
            ("cic_ids2018_external", X_external, y_external),
        ):
            indices = rng.choice(len(X), min(sample_size, len(X)), replace=False)
            if model_name == "mlp":
                indices = indices[: min(200, len(indices))]
                background_indices = rng.choice(len(X_internal), 100, replace=False)
                background = X_internal[background_indices]
                explainer = shap.Explainer(
                    pipeline.predict_proba,
                    background,
                    algorithm="permutation",
                    feature_names=features,
                )
                explanation = explainer(X[indices], max_evals=2 * len(features) + 1)
                values = np.asarray(explanation.values)
                if values.ndim == 3:
                    values = values[:, :, 1]
                transformed = X[indices]
                selected = np.asarray(features)
            else:
                transformed = transform.transform(X[indices])
                explainer = shap.TreeExplainer(estimator)
                values = explainer.shap_values(transformed)
                if isinstance(values, list):
                    values = values[-1]
                values = np.asarray(values)
                if values.ndim == 3:
                    values = values[:, :, -1]
            importance = np.mean(np.abs(values), axis=0)
            domain_importance[domain] = importance
            for feature, value in zip(selected, importance):
                frames.append(
                    {
                        "model": model_name,
                        "dataset": domain,
                        "feature": feature,
                        "mean_abs_shap": float(value),
                    }
                )
            if domain == "cicids2017_internal":
                shap.summary_plot(
                    values,
                    transformed,
                    feature_names=selected,
                    show=False,
                    max_display=10,
                    plot_size=(6.5, 4.0),
                )
                fig = plt.gcf()
                fig.suptitle(f"Global SHAP — {MODEL_LABELS[model_name]}")
                _save(fig, f"shap_summary_{model_name}")
        if len(domain_importance) == 2:
            correlation = spearmanr(
                domain_importance["cicids2017_internal"],
                domain_importance["cic_ids2018_external"],
            ).statistic
            write_json(
                resolve(f"artifacts/results/shap_stability_{model_name}.json"),
                {"model": model_name, "spearman_rank_correlation": float(correlation)},
            )
    table = pd.DataFrame(frames)
    table["rank"] = table.groupby(["model", "dataset"]).mean_abs_shap.rank(
        ascending=False, method="min"
    )
    table.to_csv(resolve("artifacts/tables/shap_feature_importance.csv"), index=False)
    return table
def error_analysis() -> pd.DataFrame:
    rows = []
    matrix_dir = resolve("artifacts/figures/confusion_matrices")
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_NAMES:
        for dataset in ("cicids2017_internal", "cic_ids2018_external"):
            path = resolve(f"artifacts/results/predictions/{model}_{dataset}_seed_42.npz")
            if not path.exists():
                continue
            data = np.load(path)
            y_true, y_pred, probability = data["y_true"], data["y_pred"], data["probability"]
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, axis = plt.subplots(figsize=(2.7, 2.35))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"],
                ax=axis,
            )
            axis.set_title(f"{MODEL_LABELS[model]} — {dataset.replace('_', ' ')}")
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            _save(fig, f"confusion_matrices/{model}_{dataset}")
            for error_type, mask in (
                ("false_positive", (y_true == 0) & (y_pred == 1)),
                ("false_negative", (y_true == 1) & (y_pred == 0)),
            ):
                values = probability[mask]
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "error_type": error_type,
                        "count": int(mask.sum()),
                        "mean_attack_probability": float(values.mean()) if len(values) else np.nan,
                        "median_attack_probability": float(np.median(values)) if len(values) else np.nan,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(resolve("artifacts/tables/error_analysis_summary.csv"), index=False)
    features = load_partition("cic_ids2018_external")[2]
    raw = pd.read_csv(
        resolve("data/processed/cic_ids2018/cic2018_aligned_features.csv"),
        usecols=features + ["label"],
        low_memory=False,
    )
    raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw.dropna(subset=features + ["label"], inplace=True)
    raw["binary_label"] = (pd.to_numeric(raw["label"]) != 0).astype("int8")
    raw.drop_duplicates(subset=features + ["binary_label"], inplace=True)
    raw.reset_index(drop=True, inplace=True)
    family_names = {0: "Benign", 1: "FTP-BruteForce", 2: "SSH-Bruteforce"}
    family_rows = []
    for model in MODEL_NAMES:
        path = resolve(
            f"artifacts/results/predictions/{model}_cic_ids2018_external_seed_42.npz"
        )
        if not path.exists():
            continue
        prediction = np.load(path)["y_pred"]
        if len(prediction) != len(raw):
            continue
        for label, group in raw.groupby("label"):
            indices = group.index.to_numpy()
            y_binary = raw.loc[indices, "binary_label"].to_numpy()
            if int(label) == 0:
                errors = (prediction[indices] == 1)
                error_name = "false_positive"
            else:
                errors = (prediction[indices] == 0)
                error_name = "false_negative"
            family_rows.append(
                {
                    "model": model,
                    "traffic_family": family_names.get(int(label), str(label)),
                    "error_type": error_name,
                    "support": len(indices),
                    "error_count": int(errors.sum()),
                    "error_rate": float(errors.mean()),
                }
            )
    pd.DataFrame(family_rows).to_csv(
        resolve("artifacts/tables/error_analysis_by_traffic_family.csv"), index=False
    )
    return table
def workflow_figure() -> None:
    fig, axis = plt.subplots(figsize=(7.16, 2.2))
    axis.axis("off")
    stages = [
        ("Datasets", "CICIDS2017\nCSE-CIC-IDS2018"),
        ("Protocol", "Deduplicate\nLeakage audit\nImmutable splits"),
        ("Models", "DT · RF · XGB\nLGBM · CatBoost · MLP"),
        ("Hybrid", "Classifier\nConfidence router\nAutoencoder"),
        ("Analyses", "10 seeds · Statistics\nCalibration · SHAP\nDrift · Efficiency"),
        ("Artifacts", "CSV · XLSX\nPDF/PNG · Models\nManifests"),
    ]
    positions = np.linspace(0.07, 0.93, len(stages))
    for index, (title, body) in enumerate(stages):
        axis.text(
            positions[index],
            0.55,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF2F8", edgecolor="#2E5D7B"),
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(positions[index + 1] - 0.075, 0.55),
                xytext=(positions[index] + 0.075, 0.55),
                arrowprops=dict(arrowstyle="->", color="#333333"),
            )
    axis.set_title("Leakage-Safe Hybrid IDS Benchmark Workflow", fontweight="bold")
    _save(fig, "overall_benchmark_workflow")
def excel_workbook() -> None:
    output = resolve("artifacts/tables/benchmark_tables.xlsx")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for path in sorted(resolve("artifacts/tables").glob("*.csv")):
            sheet = path.stem[:31]
            pd.read_csv(path).to_excel(writer, sheet_name=sheet, index=False)
def run_all(include_shap: bool = True) -> None:
    _style()
    summary_tables()
    performance_figure()
    hybrid_ablation_figure()
    routing_sensitivity()
    calibration_figures()
    drift_analysis()
    error_analysis()
    statistical_tests()
    workflow_figure()
    if include_shap:
        shap_analysis()
    excel_workbook()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    run_all(include_shap=not args.skip_shap)
if __name__ == "__main__":
    main()
