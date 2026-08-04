                                                                          

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

from benchmark.common import resolve, write_json
from benchmark.readiness_final import paths
from benchmark.readiness_supervised import load_data


def dataset_statistics() -> pd.DataFrame:
    output = paths()
    rows = []
    for name in ("train", "validation", "test"):
        X, y, family, _ = load_data(name)
        rows.append({
            "dataset": "CICIDS2017", "partition": name, "rows": len(y),
            "benign": int((y == 0).sum()), "attack": int((y == 1).sum()),
            "attack_families": int(pd.Series(family[y == 1]).nunique()),
            "features": X.shape[1],
        })
    for filename, dataset in (
        ("cse_cic_ids2018_duplicate_audit.csv", "CSE-CIC-IDS2018"),
        ("cic_unsw_dataset_statistics.csv", "CIC-UNSW-NB15"),
    ):
        source = pd.read_csv(output["tables"] / filename)
        rows.append({
            "dataset": dataset, "partition": "external_test",
            "rows": int(source["rows_after"].sum() if "rows_after" in source else source["rows_after_dedup"].sum()),
            "benign": int(source.iloc[0]["rows_after"] if "rows_after" in source else source.iloc[0]["rows_after_dedup"]),
            "attack": int(
                (source["rows_after"].sum() - source.iloc[0]["rows_after"])
                if "rows_after" in source
                else (source["rows_after_dedup"].sum() - source.iloc[0]["rows_after_dedup"])
            ),
            "attack_families": len(source) - 1, "features": 10,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "dataset_statistics.csv", index=False)
    return table


def ablation() -> pd.DataFrame:
    output = paths()
    supervised = pd.read_csv(output["results"] / "final_supervised_runs.csv")
    hybrid = pd.read_csv(output["results"] / "final_hybrid_runs.csv")
    anomaly = pd.read_csv(output["results"] / "final_anomaly_runs.csv")
    rows = []
    for dataset in supervised["dataset"].unique():
        rf = supervised[
            (supervised["dataset"] == dataset) & (supervised["model"] == "random_forest")
        ]
        hy = hybrid[hybrid["dataset"] == dataset]
        an = anomaly[
            (anomaly["dataset"] == dataset)
            & (anomaly["anomaly_model"] == "isolation_forest")
        ]
        for component, frame in (
            ("supervised_backbone", rf),
            ("anomaly_only", an),
            ("calibrated_frozen_fusion", hy),
        ):
            rows.append({
                "dataset": dataset, "component": component,
                **{f"{metric}_mean": frame[metric].mean()
                   for metric in ("mcc", "pr_auc", "tpr", "fpr", "ece", "brier")},
            })
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "component_ablation.csv", index=False)
    return table


def figures() -> None:
    output = paths()
                                 
    labels = [
        "Real datasets\n+ provenance", "Deduplicate\n+ align features",
        "Train/validation\nselection only", "Freeze calibration,\nanomaly & fusion",
        "10-seed internal\n+ external tests", "Statistics, SHAP,\ndrift & efficiency",
    ]
    fig, axis = plt.subplots(figsize=(7.16, 1.8))
    axis.set_xlim(0, 12); axis.set_ylim(0, 2); axis.axis("off")
    for i, label in enumerate(labels):
        x = 0.1 + i * 2
        box = FancyBboxPatch(
            (x, 0.55), 1.65, 0.8, boxstyle="round,pad=0.05",
            facecolor="#e8f1fa", edgecolor="#255f85", linewidth=0.9,
        )
        axis.add_patch(box)
        axis.text(x + 0.825, 0.95, label, ha="center", va="center", fontsize=7)
        if i < len(labels) - 1:
            axis.annotate("", xy=(x + 1.98, 0.95), xytext=(x + 1.68, 0.95),
                          arrowprops={"arrowstyle": "->", "lw": 0.8})
    fig.savefig(output["figures"] / "overall_benchmark_workflow.png", dpi=600)
    plt.close(fig)
                                                            
    calibration = pd.read_csv(output["tables"] / "calibration_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    for axis, dataset in zip(axes, calibration["dataset"].unique()):
        part = calibration[calibration["dataset"] == dataset]
        x = np.arange(len(part))
        axis.bar(x - .18, part["ece_mean"], .36, label="ECE")
        axis.bar(x + .18, part["brier_mean"], .36, label="Brier")
        axis.set_xticks(x, part["model"].str.replace("_", " "), rotation=70, ha="right")
        axis.set_title(dataset.replace("_", " "))
    axes[0].set_ylabel("Mean calibration error")
    axes[-1].legend(frameon=False)
    fig.savefig(output["figures"] / "calibration_comparison.png", dpi=600)
    plt.close(fig)


def inventory() -> pd.DataFrame:
    output = paths()
    rows = []
    for folder in ("tables", "figures", "manifests", "results"):
        for path in sorted(output[folder].rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            rows.append({
                "category": folder, "path": str(path.relative_to(output["root"])),
                "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output["root"] / "ARTIFACT_INVENTORY.csv", index=False)
    return table


def claims_audit() -> dict:
    output = paths()
    verdict = json.loads((output["manifests"] / "final_readiness_verdict.json").read_text())
    unknown = pd.read_csv(output["results"] / "unknown_family_runs.csv")
    comparison = unknown.pivot_table(
        index=["seed", "held_out_family"], columns="variant", values="mcc"
    )
    family_fraction = float(
        (
            comparison.groupby("held_out_family").mean()["frozen_hybrid"]
            > comparison.groupby("held_out_family").mean()["classifier_only"]
        ).mean()
    )
    audit = {
        "real_data_only": True,
        "external_label_retuning": False,
        "feature_overlap_train_validation_test": 0,
        "temporal_claim_removed": True,
        "eligible_unknown_families": int(
            unknown["held_out_family"].nunique()
        ),
        "unknown_families_mcc_improved_fraction": family_fraction,
        "positive_superiority_claim_allowed": False,
        "negative_robustness_claim_allowed": True,
        "verdict": verdict,
    }
    write_json(output["manifests"] / "claims_audit.json", audit)
    return audit


def update_workbook(dataset_table: pd.DataFrame, ablation_table: pd.DataFrame) -> None:
    output = paths()
    workbook = output["tables"] / "manuscript_results.xlsx"
    additions = {
        "dataset_statistics": dataset_table,
        "component_ablation": ablation_table,
        "unknown_family": pd.read_csv(output["tables"] / "unknown_family_mean_std.csv"),
        "anomaly_baselines": pd.read_csv(output["tables"] / "anomaly_baseline_mean_std.csv"),
        "cross_domain_shap": pd.read_csv(output["tables"] / "cross_dataset_shap_comparison.csv"),
    }
    with pd.ExcelWriter(
        workbook, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        for sheet, table in additions.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)


def main() -> None:
    dataset_table = dataset_statistics()
    ablation_table = ablation()
    figures()
    claims_audit()
    update_workbook(dataset_table, ablation_table)
    inventory()


if __name__ == "__main__":
    main()
