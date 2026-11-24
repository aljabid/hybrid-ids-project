from __future__ import annotations
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon
from sklearn.metrics import ConfusionMatrixDisplay
from benchmark.common import resolve, write_json
from benchmark.metrics import expected_calibration_error
from benchmark.readiness_final import load_external, paths
from benchmark.readiness_supervised import load_data
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 150,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})
FAMILY_NAMES = {
    "cicids2017_internal": {
        "BENIGN": "Benign",
    },
    "cic_ids2018_external": {0: "Benign", 1: "FTP-BruteForce", 2: "SSH-Bruteforce"},
    "cic_unsw_nb15_external": {
        0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
        5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
    },
}
def _runs() -> pd.DataFrame:
    root = paths()["results"]
    return pd.concat(
        [
            pd.read_csv(root / "final_supervised_runs.csv"),
            pd.read_csv(root / "final_hybrid_runs.csv"),
        ],
        ignore_index=True,
    )
def summary_tables() -> dict[str, pd.DataFrame]:
    output = paths()
    runs = _runs()
    metrics = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc",
        "tpr", "fpr", "ece", "brier", "negative_log_likelihood",
    ]
    grouped = runs.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped.reset_index(inplace=True)
    grouped.to_csv(output["tables"] / "benchmark_mean_std.csv", index=False)
    formatted = runs.groupby(["dataset", "model"])[metrics].agg(
        lambda x: f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"
    ).reset_index()
    formatted.to_csv(output["tables"] / "benchmark_formatted.csv", index=False)
    calibration = grouped[
        ["dataset", "model", "ece_mean", "ece_std", "brier_mean", "brier_std",
         "negative_log_likelihood_mean", "negative_log_likelihood_std"]
    ]
    calibration.to_csv(output["tables"] / "calibration_results.csv", index=False)
    efficiency = runs.groupby(["dataset", "model"])[
        ["training_seconds", "inference_seconds", "inference_us_per_flow",
         "peak_memory_mb", "model_bytes"]
    ].agg(["mean", "std"])
    efficiency.columns = [f"{a}_{b}" for a, b in efficiency.columns]
    efficiency.reset_index(inplace=True)
    efficiency.to_csv(output["tables"] / "efficiency_results.csv", index=False)
    return {"benchmark": grouped, "calibration": calibration, "efficiency": efficiency}
def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()
def statistical_tests() -> pd.DataFrame:
    output = paths()
    runs = _runs()
    rows = []
    for dataset in runs["dataset"].unique():
        for metric in ("mcc", "pr_auc", "tpr", "fpr", "ece", "brier"):
            subset = runs[
                (runs["dataset"] == dataset)
                & runs["model"].isin(["random_forest", "frozen_hybrid"])
            ].pivot(index="seed", columns="model", values=metric).dropna()
            difference = subset["frozen_hybrid"] - subset["random_forest"]
            try:
                statistic, pvalue = wilcoxon(difference)
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            rows.append({
                "dataset": dataset, "metric": metric, "n_pairs": len(difference),
                "rf_mean": subset["random_forest"].mean(),
                "hybrid_mean": subset["frozen_hybrid"].mean(),
                "hybrid_minus_rf": difference.mean(),
                "wilcoxon_statistic": statistic, "p_raw": pvalue,
            })
    table = pd.DataFrame(rows)
    table["p_holm"] = _holm(table["p_raw"].tolist())
    table["significant_0_05"] = table["p_holm"] <= 0.05
    table.to_csv(output["tables"] / "statistical_tests_wilcoxon_holm.csv", index=False)
    return table
def family_error_tables() -> pd.DataFrame:
    output = paths()
    rows = []
    for archive in sorted((output["results"] / "predictions").glob("*_seed_42.npz")):
        name = archive.stem
        dataset = next(d for d in FAMILY_NAMES if d in name)
        model = name[: -(len(dataset) + len("_seed_42") + 1)]
        data = np.load(archive)
        for family in np.unique(data["family"]):
            mask = data["family"] == family
            y, pred = data["y_true"][mask], data["prediction"][mask]
            family_key = family.item() if hasattr(family, "item") else family
            rows.append({
                "dataset": dataset, "model": model, "family_id": str(family_key),
                "family": FAMILY_NAMES.get(dataset, {}).get(family_key, str(family_key)),
                "n": int(mask.sum()), "false_positives": int(((y == 0) & (pred == 1)).sum()),
                "false_negatives": int(((y == 1) & (pred == 0)).sum()),
                "family_tpr": float((pred[y == 1] == 1).mean()) if (y == 1).any() else np.nan,
            })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "family_error_analysis_seed_42.csv", index=False)
    return table
def reliability_and_confusion() -> None:
    output = paths()
    for dataset in FAMILY_NAMES:
        archives = sorted((output["results"] / "predictions").glob(f"*_{dataset}_seed_42.npz"))
        fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.2))
        for axis, archive in zip(axes.ravel(), archives):
            data = np.load(archive)
            _, bins = expected_calibration_error(data["y_true"], data["probability"])
            axis.plot([0, 1], [0, 1], "--", color="0.5", lw=0.8)
            axis.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
                      marker="o", ms=2, lw=1)
            axis.set_title(archive.stem.split(f"_{dataset}")[0].replace("_", " "))
            axis.set_xlim(0, 1); axis.set_ylim(0, 1)
        for axis in axes.ravel()[len(archives):]:
            axis.axis("off")
        fig.supxlabel("Predicted confidence"); fig.supylabel("Empirical accuracy")
        fig.savefig(output["figures"] / f"reliability_{dataset}.png")
        plt.close(fig)
    for archive in sorted((output["results"] / "predictions").glob("*_seed_42.npz")):
        data = np.load(archive)
        fig, axis = plt.subplots(figsize=(2.6, 2.3))
        ConfusionMatrixDisplay.from_predictions(
            data["y_true"], data["prediction"], labels=[0, 1],
            display_labels=["Benign", "Attack"], cmap="Blues", colorbar=False, ax=axis,
        )
        axis.set_title(archive.stem.replace("_", " "))
        fig.savefig(output["figures"] / f"confusion_{archive.stem}.png")
        plt.close(fig)
def drift() -> pd.DataFrame:
    output = paths()
    X_train, _, _, features = load_data("train")
    datasets = {
        "CSE-CIC-IDS2018": load_external()["cic_ids2018_external"][0],
        "CIC-UNSW-NB15": load_external()["cic_unsw_nb15_external"][0],
    }
    rows = []
    rng = np.random.default_rng(42)
    base = X_train[rng.choice(len(X_train), min(200000, len(X_train)), replace=False)]
    for domain, target_all in datasets.items():
        target = target_all[
            rng.choice(len(target_all), min(200000, len(target_all)), replace=False)
        ]
        for index, feature in enumerate(features):
            combined = np.concatenate([base[:, index], target[:, index]])
            edges = np.unique(np.quantile(combined, np.linspace(0, 1, 51)))
            if len(edges) < 3:
                jsd = 0.0
            else:
                p = np.histogram(base[:, index], bins=edges)[0] + 1e-8
                q = np.histogram(target[:, index], bins=edges)[0] + 1e-8
                jsd = float(jensenshannon(p / p.sum(), q / q.sum()) ** 2)
            rows.append({"dataset": domain, "feature": feature, "js_divergence": jsd})
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "feature_drift_jsd.csv", index=False)
    matrix = table.pivot(index="feature", columns="dataset", values="js_divergence")
    fig, axis = plt.subplots(figsize=(3.5, 3.3))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="magma", ax=axis)
    axis.set_title("Feature drift (Jensen–Shannon divergence)")
    fig.savefig(output["figures"] / "feature_drift_heatmap.png")
    plt.close(fig)
    return table
def performance_plot(summary: pd.DataFrame) -> None:
    output = paths()
    order = ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost", "mlp", "frozen_hybrid"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    for axis, dataset in zip(axes, summary["dataset"].unique()):
        part = summary[summary["dataset"] == dataset].set_index("model").reindex(order)
        axis.barh(np.arange(len(part)), part["mcc_mean"], xerr=part["mcc_std"], capsize=2)
        axis.axvline(0, color="black", lw=0.5)
        axis.set_title(dataset.replace("_", " "))
        axis.set_yticks(np.arange(len(part)), [x.replace("_", " ") for x in part.index])
        axis.set_xlabel("MCC (mean ± SD)")
    fig.savefig(output["figures"] / "performance_comparison.png")
    plt.close(fig)
def shap_tables() -> pd.DataFrame:
    output = paths()
    X_test, _, _, features = load_data("test")
    rng = np.random.default_rng(42)
    explain_X = X_test[rng.choice(len(X_test), 400, replace=False)]
    rows = []
    for model_name in ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]:
        with (output["models"] / f"{model_name}_seed_42.pkl").open("rb") as handle:
            pipeline = pickle.load(handle)
        transformed = pipeline["scaler"].transform(explain_X)
        explainer = shap.TreeExplainer(pipeline["model"])
        values = np.asarray(explainer.shap_values(transformed))
        if values.ndim == 3:
            values = values[:, :, -1]
        elif values.ndim == 3 and values.shape[0] == 2:
            values = values[-1]
        importance = np.abs(values).mean(axis=0)
        for feature, value in zip(features, importance):
            rows.append({"model": model_name, "feature": feature, "mean_abs_shap": float(value)})
    with (output["models"] / "mlp_seed_42.pkl").open("rb") as handle:
        mlp = pickle.load(handle)
    background = X_test[rng.choice(len(X_test), 50, replace=False)]
    mlp_explainer = shap.Explainer(
        lambda value: mlp.predict_proba(value)[:, 1],
        background,
        algorithm="permutation",
        feature_names=features,
    )
    mlp_values = mlp_explainer(explain_X[:100], max_evals=2 * len(features) + 1)
    for feature, value in zip(features, np.abs(mlp_values.values).mean(axis=0)):
        rows.append({"model": "mlp", "feature": feature, "mean_abs_shap": float(value)})
    rf_rows = [row for row in rows if row["model"] == "random_forest"]
    for row in rf_rows:
        rows.append({**row, "model": "frozen_hybrid_backbone"})
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("model")["mean_abs_shap"].rank(
        ascending=False, method="first"
    ).astype(int)
    table.sort_values(["model", "rank"]).to_csv(
        output["tables"] / "shap_feature_ranking.csv", index=False
    )
    top = table[table["rank"] <= 10]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8))
    for axis, (model, part) in zip(axes.ravel(), top.groupby("model")):
        part = part.sort_values("mean_abs_shap")
        axis.barh(part["feature"], part["mean_abs_shap"])
        axis.set_title(model.replace("_", " "))
        axis.set_xlabel("mean |SHAP|")
    for axis in axes.ravel()[top["model"].nunique():]:
        axis.axis("off")
    fig.savefig(output["figures"] / "global_shap_summary.png")
    plt.close(fig)
    frozen = json.loads((output["manifests"] / "method_freeze.json").read_text())
    pd.DataFrame({
        "fusion_input": ["classifier_probability", "anomaly_probability", "classifier_entropy"],
        "logistic_coefficient": frozen["fusion"]["parameters"]["coefficients"],
    }).to_csv(output["tables"] / "hybrid_fusion_input_importance.csv", index=False)
    return table
def cross_domain_shap() -> pd.DataFrame:
    output = paths()
    _, _, _, features = load_data("test")
    domains = {
        "cicids2017_internal": load_data("test")[0],
        **{name: value[0] for name, value in load_external().items()},
    }
    with (output["models"] / "random_forest_seed_42.pkl").open("rb") as handle:
        pipeline = pickle.load(handle)
    explainer = shap.TreeExplainer(pipeline["model"])
    rng = np.random.default_rng(42)
    rows = []
    for domain, X in domains.items():
        sample = X[rng.choice(len(X), min(400, len(X)), replace=False)]
        values = np.asarray(explainer.shap_values(pipeline["scaler"].transform(sample)))
        if values.ndim == 3:
            values = values[:, :, -1]
        for feature, importance in zip(features, np.abs(values).mean(axis=0)):
            rows.append({
                "dataset": domain, "model": "random_forest",
                "feature": feature, "mean_abs_shap": float(importance),
            })
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("dataset")["mean_abs_shap"].rank(
        ascending=False, method="first"
    ).astype(int)
    table.to_csv(output["tables"] / "cross_dataset_shap_comparison.csv", index=False)
    return table
def verdict(tests: pd.DataFrame) -> dict:
    output = paths()
    runs = _runs()
    comparisons = runs[
        runs["model"].isin(["random_forest", "frozen_hybrid"])
        & runs["dataset"].str.contains("external")
    ].groupby(["dataset", "model"])[["mcc", "tpr", "fpr"]].mean().unstack("model")
    gates = {}
    for dataset, row in comparisons.iterrows():
        gates[dataset] = {
            "mcc_improvement": float(row[("mcc", "frozen_hybrid")] - row[("mcc", "random_forest")]),
            "tpr_improvement": float(row[("tpr", "frozen_hybrid")] - row[("tpr", "random_forest")]),
            "fpr_increase": float(row[("fpr", "frozen_hybrid")] - row[("fpr", "random_forest")]),
        }
    result = {
        "positive_superiority_manuscript_ready": False,
        "reason": (
            "Frozen hybrid fails the predeclared external MCC/TPR improvement "
            "gates and cross-domain attack recall is operationally unacceptable."
        ),
        "external_gates": gates,
        "external_labels_used_for_retuning": False,
        "appropriate_claim": (
            "Negative robustness study: strong in-domain scores do not transfer "
            "under severe semantic/covariate shift."
        ),
    }
    write_json(output["manifests"] / "final_readiness_verdict.json", result)
    return result
def excel_export(tables: dict[str, pd.DataFrame], stats: pd.DataFrame,
                 errors: pd.DataFrame, drift_table: pd.DataFrame,
                 shap_table: pd.DataFrame) -> None:
    target = paths()["tables"] / "manuscript_results.xlsx"
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
        stats.to_excel(writer, sheet_name="statistical_tests", index=False)
        errors.to_excel(writer, sheet_name="family_errors", index=False)
        drift_table.to_excel(writer, sheet_name="feature_drift", index=False)
        shap_table.to_excel(writer, sheet_name="shap_ranking", index=False)
        pd.read_csv(paths()["tables"] / "supervised_tuning_summary.csv").to_excel(
            writer, sheet_name="hyperparameters", index=False
        )
def main() -> None:
    tables = summary_tables()
    stats = statistical_tests()
    errors = family_error_tables()
    reliability_and_confusion()
    drift_table = drift()
    performance_plot(tables["benchmark"])
    shap_table = shap_tables()
    cross_domain_shap()
    excel_export(tables, stats, errors, drift_table, shap_table)
    print(json.dumps(verdict(stats), indent=2))
if __name__ == "__main__":
    main()
