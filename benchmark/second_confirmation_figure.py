from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark.redesign_preprocessing import paths
VARIANT_LABELS = {
    "adaptive_decision_tree": "Adaptive DT branch\n(unlabeled threshold)",
    "selective_agreement_hybrid": "Selective agreement hybrid\n(deployed, frozen method)",
}
def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.7,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
def build():
    output = paths()
    style()
    ton_boot = pd.read_csv(output["tables"] / "robustness_bootstrap_extended.csv")
    ton_boot = ton_boot[ton_boot["experiment"] == "frozen_confirmation_table_vi"]
    ton_boot = ton_boot.rename(columns={"variant": "variant"})
    ton_boot["dataset"] = "CIC-ToN-IoT\n(frozen confirmation)"
    ddos_boot = pd.read_csv(output["tables"] / "cicddos2019_second_confirmation_bootstrap_ci.csv")
    ddos_boot["dataset"] = "CIC-DDoS2019\n(second confirmation)"
    combined = pd.concat([
        ton_boot[["dataset", "variant", "metric", "mean", "ci95_low", "ci95_high"]],
        ddos_boot[["dataset", "variant", "metric", "mean", "ci95_low", "ci95_high"]],
    ], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), constrained_layout=True)
    color_map = {"CIC-ToN-IoT\n(frozen confirmation)": "#CC79A7",
                 "CIC-DDoS2019\n(second confirmation)": "#0072B2"}
    slots = {
        ("adaptive_decision_tree", "CIC-ToN-IoT\n(frozen confirmation)"): 0,
        ("adaptive_decision_tree", "CIC-DDoS2019\n(second confirmation)"): 1,
        ("selective_agreement_hybrid", "CIC-ToN-IoT\n(frozen confirmation)"): 2.6,
        ("selective_agreement_hybrid", "CIC-DDoS2019\n(second confirmation)"): 3.6,
    }
    for ax, metric, title in (
        (axes[0], "mcc", "(A) MCC, seed-level bootstrap 95% CI"),
        (axes[1], "roc_auc", "(B) ROC-AUC, seed-level bootstrap 95% CI"),
    ):
        part = combined[combined["metric"] == metric]
        for (variant, dataset), pos in slots.items():
            row = part[(part["variant"] == variant) & (part["dataset"] == dataset)]
            if row.empty:
                continue
            row = row.iloc[0]
            lo = row["mean"] - row["ci95_low"]
            hi = row["ci95_high"] - row["mean"]
            ax.errorbar(pos, row["mean"], yerr=[[lo], [hi]], fmt="o",
                        color=color_map[dataset], markersize=5, capsize=3, linewidth=1.2)
        ax.set_xlim(-0.6, 4.2)
        ax.set_xticks([0.5, 3.1])
        ax.set_xticklabels([VARIANT_LABELS["adaptive_decision_tree"],
                             VARIANT_LABELS["selective_agreement_hybrid"]])
        ax.axhline(0 if metric == "mcc" else 0.5, color="black", linewidth=0.5, linestyle=":")
        ax.set_ylabel(metric.upper() if metric == "mcc" else "ROC-AUC")
        ax.set_title(title)
        if metric == "roc_auc":
            ax.text(0.02, 0.02, "ToN-IoT ROC-AUC not in bootstrap table (pr_auc only)",
                    transform=ax.transAxes, fontsize=5.8, color="#666666", va="bottom")
    handles = [
        plt.Line2D([0], [0], marker="o", color="#CC79A7", linestyle="", label="CIC-ToN-IoT (frozen)"),
        plt.Line2D([0], [0], marker="o", color="#0072B2", linestyle="", label="CIC-DDoS2019 (second)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.13))
    fig.suptitle("Two independent external confirmations of the frozen method", y=1.22, fontsize=9.5)
    fig.savefig(output["figures"] / "second_confirmation_comparison.png")
    fig.savefig(output["figures"] / "second_confirmation_comparison.pdf")
    plt.close(fig)
if __name__ == "__main__":
    build()
    print("Figure written: second_confirmation_comparison.{png,pdf}")
