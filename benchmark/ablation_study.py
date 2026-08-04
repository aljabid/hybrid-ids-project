"""Controlled, single-variable ablation study (post-manuscript addendum).

Five candidate axes were specified: (1) random-row vs. group-aware split,
(2) without vs. with held-out-family exclusion, (3) without vs. with
post-hoc calibration, (4) without vs. with the frozen confirmation
protocol, (5) without vs. with semantic feature-schema alignment.

Axes 1-4 are answered here, each changing exactly one protocol component
while holding everything else (data, hyperparameters, seeds, models)
identical, and each reusing already-computed, already-checksummed
artifacts -- no model is retrained for axes 1-3, and axis 4 adds exactly
one new, explicitly-labeled, non-deployable diagnostic computed on the
already-frozen seed-42 confirmation probabilities (no retraining, no
change to the actual frozen decision rule or its reported numbers).

Axis 5 (semantic alignment) is intentionally NOT executed. Removing
semantic alignment does not hold "everything else identical": every
domain's raw export uses a different, non-comparable column set and
dimensionality, so "without alignment" has no well-defined fixed-length
input a frozen model could score -- the only way to test it is to train a
new model on a different, ad hoc feature space, which is a new modeling
exercise, not a controlled ablation, and risks the exact kind of
uncontrolled redesign the task instructions rule out. This is recorded
explicitly rather than silently skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from benchmark.common import write_json
from benchmark.metrics import binary_metrics
from benchmark.redesign_preprocessing import paths

FAMILY_MODELS = ["family_aware_rf", "redesigned_meta_hybrid"]


def _agg(df, metrics, prefix):
    out = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        out[f"{prefix}_{metric}_mean"] = float(df[metric].mean())
        out[f"{prefix}_{metric}_std"] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
    out[f"{prefix}_n"] = int(len(df))
    return out


def axis_split_strategy():
    """Axis 1: random-row vs. group-aware split, all 8 mandatory models."""
    output = paths()
    grouped = pd.read_csv(output["results"] / "grouped_ten_seed_runs.csv")
    group_aware = grouped[grouped["dataset"] == "cicids2017_group_test"]
    random_a = pd.read_csv(output["results"] / "matched_random_row_ten_seed_runs.csv")
    random_b = pd.read_csv(output["results"] / "matched_random_row_extended_ten_seed_runs.csv")
    random_row = pd.concat([random_a, random_b], ignore_index=True)

    metrics = ["mcc", "roc_auc", "ece", "brier"]
    rows = []
    for model in sorted(group_aware["model"].unique()):
        ga = group_aware[group_aware["model"] == model]
        rr = random_row[random_row["model"] == model]
        if ga.empty or rr.empty:
            continue
        row = {"axis": "split_strategy", "component_variant": model}
        row.update(_agg(ga, metrics, "group_aware"))
        row.update(_agg(rr, metrics, "random_row"))
        row["mcc_delta_random_minus_group"] = (
            row["random_row_mcc_mean"] - row["group_aware_mcc_mean"]
        )
        row["recovery"] = ""
        row["coverage"] = ""
        row["note"] = "Not applicable: no rejector/coverage in this axis."
        rows.append(row)
    return pd.DataFrame(rows)


def axis_family_exclusion():
    """Axis 2: without vs. with held-out-family exclusion, family_aware_rf
    and redesigned_meta_hybrid (the two models the held-out-family study
    was run on).
    """
    output = paths()
    grouped = pd.read_csv(output["results"] / "grouped_ten_seed_runs.csv")
    without = grouped[grouped["dataset"] == "cicids2017_group_test"]
    with_excl = pd.read_csv(output["results"] / "unknown_family_ten_seed_runs.csv")

    metrics = ["mcc", "roc_auc", "ece", "brier", "tpr"]
    rows = []
    for model in FAMILY_MODELS:
        wo = without[without["model"] == model]
        wi = with_excl[with_excl["variant"] == model]
        row = {"axis": "family_exclusion", "component_variant": model}
        row.update(_agg(wo, metrics, "without_exclusion"))
        row.update(_agg(wi, metrics, "with_exclusion"))
        row["mcc_delta_with_minus_without"] = (
            row["with_exclusion_mcc_mean"] - row["without_exclusion_mcc_mean"]
        )
        row["recovery"] = ""
        row["coverage"] = ""
        row["note"] = (
            "with_exclusion pools all 5 eligible held-out families "
            f"({sorted(with_excl['held_out_family'].unique().tolist())}) x 10 seeds."
        )
        rows.append(row)
    return pd.DataFrame(rows)


def axis_calibration():
    """Axis 3: without vs. with post-hoc temperature-scaled calibration."""
    output = paths()
    recal = pd.read_csv(output["tables"] / "recalibration_seed42.csv")
    rows = []
    for _, r in recal.iterrows():
        rows.append({
            "axis": "calibration",
            "component_variant": f"{r['model']}/{r['domain']}",
            "without_calibration_mcc_mean": r["baseline_mcc"],
            "with_calibration_mcc_mean": r["temperature_scaled_mcc"],
            "mcc_delta_with_minus_without": r["temperature_scaled_mcc"] - r["baseline_mcc"],
            "without_calibration_ece_mean": r["baseline_ece"],
            "with_calibration_ece_mean": r["temperature_scaled_ece"],
            "without_calibration_brier_mean": r["baseline_brier"],
            "with_calibration_brier_mean": r["temperature_scaled_brier"],
            "recovery": "", "coverage": "",
            "note": (
                "Temperature scaling is a monotonic transform of the score, "
                "so MCC/TPR/FPR are unchanged by construction (delta=0 "
                "confirmed exactly); only ECE/Brier/NLL move. Seed 42 only, "
                "same scope as recalibration_seed42.csv."
            ),
        })
    return pd.DataFrame(rows)


def _oracle_threshold_mcc(y, probability, grid_points=197):
    y = np.asarray(y, dtype=int)
    best_mcc, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.01, 0.99, grid_points):
        mcc = matthews_corrcoef(y, (probability >= threshold).astype(int))
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, float(threshold)
    return best_mcc, best_threshold


def axis_frozen_confirmation(force=False):
    """Axis 4: with vs. without the frozen (unlabeled, pre-registered)
    confirmation threshold.

    'With frozen protocol' is the actual, already-reported adaptive
    decision-tree branch performance at seed 42 (dt_threshold selected
    without ever looking at confirmation labels).

    'Without frozen protocol' is a diagnostic upper bound only: the best
    achievable MCC if the decision threshold were instead chosen by
    sweeping against confirmation's own labels (the same
    _oracle_threshold_mcc diagnostic already used for the two development
    domains in threshold_sensitivity_seed42.csv, extended here to the
    confirmation domain for the first time). This is explicitly NOT a
    deployable procedure -- it uses target labels to pick the threshold --
    and does not modify the frozen model, its stored predictions, or the
    manuscript's reported confirmation numbers in any way.
    """
    output = paths()
    target = output["tables"] / "confirmation_oracle_threshold_seed42.csv"
    if target.exists() and not force:
        oracle = pd.read_csv(target)
    else:
        with np.load(output["results"] / "predictions" / "confirmation_seed42.npz") as data:
            y = data["y"]
            dt_probability = data["dt_probability"]
            xgb_probability = data["xgb_probability"]
        rows = []
        for branch, probability in (("decision_tree", dt_probability), ("xgboost", xgb_probability)):
            fixed = binary_metrics(y, (probability >= 0.5).astype(int), probability)
            oracle_mcc, oracle_threshold = _oracle_threshold_mcc(y, probability)
            rows.append({
                "branch": branch,
                "roc_auc": float(roc_auc_score(y, probability)),
                "fixed_0_5_threshold_mcc": fixed["mcc"],
                "oracle_threshold": oracle_threshold,
                "oracle_threshold_mcc": oracle_mcc,
            })
            print(f"confirmation oracle threshold ({branch}): "
                  f"ROC-AUC={rows[-1]['roc_auc']:.4f} "
                  f"oracle_mcc={oracle_mcc:.4f} @ t={oracle_threshold:.3f}", flush=True)
        oracle = pd.DataFrame(rows)
        oracle.to_csv(target, index=False)

    confirmation = pd.read_csv(output["results"] / "confirmation_ten_seed_runs.csv")
    frozen_seed42 = confirmation[
        (confirmation["seed"] == 42) & (confirmation["variant"] == "adaptive_decision_tree")
    ].iloc[0]
    frozen_all_seeds = confirmation[confirmation["variant"] == "adaptive_decision_tree"]["mcc"]

    dt_row = oracle[oracle["branch"] == "decision_tree"].iloc[0]
    row = {
        "axis": "frozen_confirmation",
        "component_variant": "decision_tree_branch",
        "with_frozen_protocol_mcc_seed42": float(frozen_seed42["mcc"]),
        "with_frozen_protocol_mcc_10seed_mean": float(frozen_all_seeds.mean()),
        "with_frozen_protocol_mcc_10seed_std": float(frozen_all_seeds.std(ddof=1)),
        "with_frozen_protocol_threshold_seed42": float(frozen_seed42["dt_threshold"]),
        "without_frozen_protocol_oracle_mcc_seed42": float(dt_row["oracle_threshold_mcc"]),
        "without_frozen_protocol_oracle_threshold_seed42": float(dt_row["oracle_threshold"]),
        "roc_auc_seed42": float(dt_row["roc_auc"]),
        "mcc_recoverable_by_oracle_rethresholding": (
            float(dt_row["oracle_threshold_mcc"]) - float(frozen_seed42["mcc"])
        ),
        "recovery": float(dt_row["oracle_threshold_mcc"]) - float(frozen_seed42["mcc"]),
        "coverage": 1.0,
        "note": (
            "oracle values use confirmation LABELS to pick the threshold and "
            "are a non-deployable diagnostic upper bound, not a proposed "
            "procedure; seed 42 only, consistent with existing project "
            "precedent for this diagnostic."
        ),
    }
    return pd.DataFrame([row])


def run_all():
    output = paths()
    print("=== axis 1: split strategy ===", flush=True)
    a1 = axis_split_strategy()
    print("=== axis 2: family exclusion ===", flush=True)
    a2 = axis_family_exclusion()
    print("=== axis 3: calibration ===", flush=True)
    a3 = axis_calibration()
    print("=== axis 4: frozen confirmation ===", flush=True)
    a4 = axis_frozen_confirmation(force=True)

    a1.to_csv(output["tables"] / "ablation_axis1_split_strategy.csv", index=False)
    a2.to_csv(output["tables"] / "ablation_axis2_family_exclusion.csv", index=False)
    a3.to_csv(output["tables"] / "ablation_axis3_calibration.csv", index=False)
    a4.to_csv(output["tables"] / "ablation_axis4_frozen_confirmation.csv", index=False)

    write_json(output["manifests"] / "ablation_study_manifest.json", {
        "description": (
            "Controlled single-variable ablation study, axes 1-4. Each axis "
            "changes exactly one protocol component; all other components "
            "(data, hyperparameters, seeds, models) are held identical. "
            "Axis 5 (semantic alignment) is intentionally not executed -- "
            "see module docstring in benchmark/ablation_study.py for why."
        ),
        "axes": {
            "1_split_strategy": {
                "rows": len(a1), "source": "matched split control, 8 models x 10 seeds x 2 splits",
                "new_computation": False,
            },
            "2_family_exclusion": {
                "rows": len(a2), "source": "grouped_ten_seed_runs.csv vs unknown_family_ten_seed_runs.csv",
                "new_computation": False,
            },
            "3_calibration": {
                "rows": len(a3), "source": "recalibration_seed42.csv",
                "new_computation": False,
            },
            "4_frozen_confirmation": {
                "rows": len(a4), "source": "confirmation_seed42.npz oracle-threshold diagnostic (new)",
                "new_computation": True,
                "caveat": "oracle side uses target labels; non-deployable upper bound only",
            },
            "5_semantic_alignment": {
                "executed": False,
                "reason": "no well-defined single-variable change; would require training a new model on a different feature space",
            },
        },
    })
    print("Done. See artifacts/redesign_v3/tables/ablation_axis*.csv "
          "and manifests/ablation_study_manifest.json")


if __name__ == "__main__":
    run_all()
