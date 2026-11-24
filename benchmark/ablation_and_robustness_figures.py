from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark.redesign_preprocessing import paths
MODEL_LABELS = {
    "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "family_aware_rf": "Family-aware RF", "tabular_resnet": "Tabular ResNet",
    "redesigned_meta_hybrid": "Meta hybrid",
}
MODEL_ORDER = [
    "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
    "family_aware_rf", "tabular_resnet", "redesigned_meta_hybrid",
]
def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 6.8, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "axes.linewidth": 0.7,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
def _save(fig, output, name):
    fig.savefig(output["figures"] / f"{name}.png")
    fig.savefig(output["figures"] / f"{name}.pdf")
    plt.close(fig)
def ablation_figure():
    output = paths()
    style()
    a1 = pd.read_csv(output["tables"] / "ablation_axis1_split_strategy.csv").set_index("component_variant").loc[MODEL_ORDER].reset_index()
    a2 = pd.read_csv(output["tables"] / "ablation_axis2_family_exclusion.csv")
    a3 = pd.read_csv(output["tables"] / "ablation_axis3_calibration.csv")
    a4 = pd.read_csv(output["tables"] / "ablation_axis4_frozen_confirmation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.6), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(a1))
    width = 0.35
    ax.bar(x - width / 2, a1["group_aware_mcc_mean"], width, yerr=a1["group_aware_mcc_std"],
           label="Group-aware", color="#0072B2", edgecolor="black", linewidth=0.4, capsize=2)
    ax.bar(x + width / 2, a1["random_row_mcc_mean"], width, yerr=a1["random_row_mcc_std"],
           label="Random-row", color="#D55E00", edgecolor="black", linewidth=0.4, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in a1["component_variant"]], rotation=30, ha="right")
    ax.set_ylabel("MCC")
    ax.set_title("(A) Split strategy, all 8 models")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[0, 1]
    x = np.arange(len(a2))
    ax.bar(x - width / 2, a2["without_exclusion_mcc_mean"], width, yerr=a2["without_exclusion_mcc_std"],
           label="Without exclusion", color="#0072B2", edgecolor="black", linewidth=0.4, capsize=2)
    ax.bar(x + width / 2, a2["with_exclusion_mcc_mean"], width, yerr=a2["with_exclusion_mcc_std"],
           label="With exclusion", color="#009E73", edgecolor="black", linewidth=0.4, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in a2["component_variant"]], rotation=20, ha="right")
    ax.set_ylabel("MCC")
    ax.set_title("(B) Held-out-family exclusion")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1, 0]
    labels = [f"{MODEL_LABELS[m]}\n{d.replace('cse_cic_ids2018','CSE-CIC-2018').replace('cic_unsw_nb15','CIC-UNSW-NB15')}"
              for m, d in a3["component_variant"].str.split("/", expand=True).values]
    x = np.arange(len(a3))
    ax.bar(x - width / 2, a3["without_calibration_ece_mean"], width,
           label="Without calibration", color="#999999", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, a3["with_calibration_ece_mean"], width,
           label="With calibration (T-scaled)", color="#E69F00", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.6)
    ax.set_ylabel("ECE")
    ax.set_title("(C) Calibration (MCC unchanged by construction)")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1, 1]
    row = a4.iloc[0]
    values = [row["with_frozen_protocol_mcc_seed42"], row["without_frozen_protocol_oracle_mcc_seed42"]]
    bars = ax.bar(["Frozen threshold\n(actual, seed 42)", "Oracle threshold\n(diagnostic upper bound)"],
                   values, color=["#0072B2", "#CC79A7"], edgecolor="black", linewidth=0.4, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("MCC")
    ax.set_title("(D) Frozen confirmation, DT branch (ROC-AUC=%.3f)" % row["roc_auc_seed42"])
    ax.set_ylim(0, max(values) * 1.4 + 0.01)
    fig.suptitle("Controlled single-variable ablations (axes 1–4)", fontsize=9.5, y=1.02)
    _save(fig, output, "ablation_study_summary")
def robustness_bootstrap_figure():
    output = paths()
    style()
    boot = pd.read_csv(output["tables"] / "robustness_bootstrap_extended.csv")
    split = boot[(boot["experiment"] == "matched_split_table_ii") & (boot["metric"] == "mcc")]
    split = split.pivot(index="model", columns="split_strategy", values=["mean", "ci95_low", "ci95_high"])
    split = split.loc[[m for m in MODEL_ORDER if m in split.index]]
    conf = boot[(boot["experiment"] == "frozen_confirmation_table_vi") & (boot["metric"] == "mcc")]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(split))
    for offset, strategy, color in ((-0.18, "group_aware", "#0072B2"), (0.18, "random_row", "#D55E00")):
        means = split[("mean", strategy)].to_numpy()
        lo = means - split[("ci95_low", strategy)].to_numpy()
        hi = split[("ci95_high", strategy)].to_numpy() - means
        ax.errorbar(x + offset, means, yerr=[lo, hi], fmt="o", color=color, markersize=3.5,
                    capsize=2.5, linewidth=1, label=strategy.replace("_", "-"))
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in split.index], rotation=30, ha="right")
    ax.set_ylabel("MCC (seed-level bootstrap 95% CI)")
    ax.set_title("(A) Table II: split-strategy MCC")
    ax.legend(frameon=False)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax = axes[1]
    x = np.arange(len(conf))
    means = conf["mean"].to_numpy()
    lo = means - conf["ci95_low"].to_numpy()
    hi = conf["ci95_high"].to_numpy() - means
    ax.errorbar(x, means, yerr=[lo, hi], fmt="o", color="#CC79A7", markersize=4, capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in conf["variant"]], fontsize=6.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_ylabel("MCC (seed-level bootstrap 95% CI)")
    ax.set_title("(B) Table VI: frozen confirmation MCC")
    fig.suptitle("Seed-level bootstrap 95% confidence intervals (n=10 seeds, 20,000 resamples)",
                 fontsize=9, y=1.05)
    _save(fig, output, "robustness_bootstrap_ci")
if __name__ == "__main__":
    ablation_figure()
    robustness_bootstrap_figure()
    print("Figures written: ablation_study_summary.{png,pdf}, robustness_bootstrap_ci.{png,pdf}")
