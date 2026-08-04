from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmark.metrics import expected_calibration_error
from benchmark.redesign_preprocessing import paths


MODELS = ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
FAMILY = {
    0: "BENIGN", 1: "Bot", 2: "DDoS", 3: "DoS GoldenEye", 4: "DoS Hulk",
    5: "DoS Slowhttptest", 6: "DoS slowloris", 7: "FTP-Patator", 8: "Heartbleed",
    9: "Infiltration", 10: "PortScan", 11: "SSH-Patator",
    12: "Web Attack Brute Force", 13: "Web Attack Sql Injection", 14: "Web Attack XSS",
}


def style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, root: Path, name: str):
    for suffix in ("pdf", "svg"):
        fig.savefig(root / f"{name}.{suffix}", bbox_inches="tight")
    fig.savefig(root / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def load_prediction(root: Path, name: str):
    data = np.load(root / name, allow_pickle=False)
    return data["y_true"], data["prediction"], data["probability"], data["family"]


def error_analysis(output):
    prediction_root = output["results"] / "predictions"
    rows = []
    for strategy, pattern in (("group_aware", "{model}_cicids2017_group_test_seed_42.npz"), ("random_row", "matched_random_row_{model}_seed_42.npz")):
        for model in MODELS:
            y, prediction, _, family = load_prediction(prediction_root, pattern.format(model=model))
            for value in np.unique(family):
                mask = family == value
                rows.append({
                    "split_strategy": strategy,
                    "model": model,
                    "family_id": int(value),
                    "family": FAMILY.get(int(value), str(value)),
                    "support": int(mask.sum()),
                    "false_positive": int(np.sum((y[mask] == 0) & (prediction[mask] == 1))),
                    "false_negative": int(np.sum((y[mask] == 1) & (prediction[mask] == 0))),
                    "true_positive": int(np.sum((y[mask] == 1) & (prediction[mask] == 1))),
                    "true_negative": int(np.sum((y[mask] == 0) & (prediction[mask] == 0))),
                })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "matched_split_seed42_family_errors.csv", index=False)
    return table


def reliability(output):
    prediction_root = output["results"] / "predictions"
    datasets = [("CICIDS2017 group", "{model}_cicids2017_group_test_seed_42.npz"), ("CSE-CIC-IDS2018", "{model}_cse_cic_ids2018_seed_42.npz"), ("CIC-UNSW-NB15", "{model}_cic_unsw_nb15_seed_42.npz")]
    rows = []
    style()
    fig, axes = plt.subplots(len(MODELS), len(datasets), figsize=(7.16, 7.5), sharex=True, sharey=True, constrained_layout=True)
    for i, model in enumerate(MODELS):
        for j, (dataset, pattern) in enumerate(datasets):
            y, prediction, probability, _ = load_prediction(prediction_root, pattern.format(model=model))
            ece, bins = expected_calibration_error(y, probability)
            axis = axes[i, j]
            axis.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=0.8)
            if bins:
                confidence = [item["confidence"] for item in bins]
                accuracy = [item["accuracy"] for item in bins]
                axis.plot(confidence, accuracy, marker="o", color="#0072B2", linewidth=1, markersize=2.5)
            axis.set_title(dataset if i == 0 else "")
            if j == 0:
                axis.set_ylabel(f"{LABELS[model]}\nAccuracy")
            if i == len(MODELS) - 1:
                axis.set_xlabel("Confidence")
            axis.text(0.04, 0.88, f"ECE={ece:.3f}", transform=axis.transAxes, fontsize=6)
            rows.append({"model": model, "dataset": dataset, "ece": ece, "n_bins_nonempty": len(bins), "seed": 42})
    save(fig, output["figures"], "reliability_mandatory_models")
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "reliability_mandatory_models_seed42.csv", index=False)
    return table


def mcc_ece(output):
    summary = pd.read_csv(output["tables"] / "matched_split_summary.csv")
    rows = []
    for _, item in summary.iterrows():
        for strategy in ("group", "random"):
            rows.append({"model": item["model"], "split_strategy": strategy, "mcc": item[f"{strategy}_mcc_mean"], "ece": item[f"{strategy}_ece_mean"]})
    table = pd.DataFrame(rows)
    rho, p_value = spearmanr(table["mcc"], table["ece"])
    table["spearman_rho_all_points"] = rho
    table["spearman_p_raw_all_points"] = p_value
    table.to_csv(output["tables"] / "matched_split_mcc_ece_exploratory.csv", index=False)
    style()
    fig, axis = plt.subplots(figsize=(3.45, 2.5), constrained_layout=True)
    for strategy, marker, color in (("group", "o", "#0072B2"), ("random", "s", "#E69F00")):
        part = table[table["split_strategy"] == strategy]
        axis.scatter(part["ece"], part["mcc"], label=strategy.replace("_", "-"), marker=marker, color=color, edgecolor="black", linewidth=0.4)
        for _, item in part.iterrows():
            axis.annotate(LABELS[item["model"]], (item["ece"], item["mcc"]), xytext=(3, 2), textcoords="offset points", fontsize=5.5)
    axis.set_xlabel("ECE")
    axis.set_ylabel("MCC")
    axis.legend(frameon=False)
    axis.grid(color="0.88", linewidth=0.5)
    save(fig, output["figures"], "matched_split_mcc_ece")
    return table


def jsd_performance(output):
    drift = pd.read_csv(output["tables"] / "feature_drift_jsd.csv")
    benchmark = pd.read_csv(output["tables"] / "benchmark_mean_std.csv")
    map_name = {"cse_cic_ids2018": "cse_cic_ids2018", "cic_unsw_nb15": "cic_unsw_nb15"}
    rows = []
    for dataset, target in map_name.items():
        mean_jsd = drift[drift["dataset"] == dataset]["js_divergence"].mean()
        part = benchmark[(benchmark["dataset"] == target) & benchmark["model"].isin(MODELS)]
        rows.append({
            "dataset": dataset, "mean_jsd": mean_jsd,
            "median_model_mcc": part["mcc_mean"].median(),
            "median_model_ece": part["ece_mean"].median(),
            "median_model_tpr": part["tpr_mean"].median(),
            "median_model_fpr": part["fpr_mean"].median(),
            "models": len(part),
        })
    confirmation = pd.read_csv(output["tables"] / "confirmation_mean_std.csv")
    rows.append({
        "dataset": "cic_ton_iot_confirmation",
        "mean_jsd": drift[drift["dataset"] == "cic_ton_iot_confirmation"]["js_divergence"].mean(),
        "median_model_mcc": np.nan,
        "median_model_ece": confirmation["ece_mean"].median(),
        "median_model_tpr": confirmation["tpr_mean"].median(),
        "median_model_fpr": confirmation["fpr_mean"].median(),
        "models": 2,
    })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "jsd_performance_exploratory.csv", index=False)
    return table


def main():
    output = paths()
    errors = error_analysis(output)
    reliability_table = reliability(output)
    calibration = mcc_ece(output)
    drift = jsd_performance(output)
    manifest = {"status": "VALIDATED", "tasks": ["focused_seed42_error_analysis", "seed42_reliability_diagrams", "mcc_ece_exploratory", "jsd_performance_descriptive"], "error_rows": len(errors), "reliability_rows": len(reliability_table), "mcc_ece_rows": len(calibration), "jsd_rows": len(drift), "limitations": ["Reliability diagrams use stored seed-42 predictions", "MCC-ECE scatter uses ten matched split/model aggregate points", "JSD-performance table has two comparable development domains and is descriptive only"]}
    (output["manifests"] / "final_publication_analysis.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
