"""Formal risk-coverage selective-classifier baseline versus the engineering
Decision Tree-XGBoost agreement rejector.

Addresses REMAINING_WEAKNESSES.md item 9: "Selective classification is
represented by one engineering rejector. Risk-coverage baselines and
formally trained selective models would be required to generalize
conclusions about abstention."

This does not retrain anything. It reuses the frozen confirmation-stage DT
and XGBoost branch probabilities already stored in
results/predictions/confirmation_seed42.npz (seed 42) and builds three
standard softmax-response (Chow's rule) selective classifiers on top of
them, in increasing order of how much information they use:

  1. dt_softmax_response       -- confidence = |dt_probability - 0.5|
  2. xgb_softmax_response      -- confidence = |xgb_probability - 0.5|
  3. disagreement_graded       -- confidence = -|dt_probability - xgb_probability|
     (a continuous generalization of the actual engineering rule, which
     only ever used the binary version of this same disagreement signal)

Each produces a full risk-coverage curve and an AURC (area under the
risk-coverage curve, trapezoidal, lower is better) by sweeping the
confidence threshold. The realized engineering rejector is plotted as a
single (coverage, risk) point on the same axes so a reader can see whether
it sits on, above, or below the achievable frontier of a standard method at
matched coverage.

Scope: confirmation domain (CIC-ToN-IoT), seed 42 only -- this is where the
engineering rejector's abstention behavior was reported and criticized in
Table V. It is not repeated across all ten seeds; see the module docstring
convention shared with benchmark.shift_and_calibration_diagnostics.
"""

from __future__ import annotations

import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.metrics import binary_metrics
from benchmark.redesign_preprocessing import paths


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, root, name):
    for suffix in ("pdf", "svg"):
        fig.savefig(root / f"{name}.{suffix}", bbox_inches="tight")
    fig.savefig(root / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _risk_coverage(y, prediction, confidence, n_points=200):
    order = np.argsort(-confidence)
    y_sorted, correct_sorted = y[order], (prediction[order] == y[order]).astype(float)
    n = len(y_sorted)
    cum_correct = np.cumsum(correct_sorted)
    coverages = np.linspace(1.0 / n, 1.0, n_points)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        risk = 1.0 - (cum_correct[k - 1] / k)
        rows.append((k / n, risk))
    return np.array(rows)


def _aurc(curve):
    coverage, risk = curve[:, 0], curve[:, 1]
    order = np.argsort(coverage)
    return float(np.trapezoid(risk[order], coverage[order]))


def run(force=False):
    output = paths()
    target = output["tables"] / "selective_classifier_risk_coverage_confirmation.csv"
    summary_target = output["tables"] / "selective_classifier_summary_confirmation.csv"
    if target.exists() and summary_target.exists() and not force:
        return pd.read_csv(target), pd.read_csv(summary_target)

    with np.load(output["results"] / "predictions" / "confirmation_seed42.npz") as data:
        y = data["y"].astype(int)
        dt_p = data["dt_probability"].astype(float)
        xgb_p = data["xgb_probability"].astype(float)
        covered = data["covered"]
        selective_prediction = data["selective_prediction"]

    dt_pred = (dt_p >= 0.5).astype(int)
    xgb_pred = (xgb_p >= 0.5).astype(int)

    strategies = {
        "dt_softmax_response": (np.abs(dt_p - 0.5), dt_pred),
        "xgb_softmax_response": (np.abs(xgb_p - 0.5), xgb_pred),
        "disagreement_graded": (-np.abs(dt_p - xgb_p), dt_pred),
    }

    curve_rows, summary_rows = [], []
    for name, (confidence, prediction) in strategies.items():
        curve = _risk_coverage(y, prediction, confidence)
        aurc = _aurc(curve)
        for coverage, risk in curve:
            curve_rows.append({"strategy": name, "coverage": coverage, "risk": risk})
        summary_rows.append({
            "strategy": name, "aurc": aurc,
            "risk_at_engineering_coverage": float(
                np.interp(float(covered.mean()), curve[:, 0], curve[:, 1])
            ),
        })

    covered_index = np.flatnonzero(covered)
    if len(covered_index) != len(selective_prediction):
        raise RuntimeError(
            f"covered mask ({len(covered_index)}) does not match "
            f"selective_prediction length ({len(selective_prediction)})"
        )
    engineering_coverage = float(covered.mean())
    engineering_risk = float(np.mean(selective_prediction != y[covered_index]))
    engineering_metrics = binary_metrics(
        y[covered_index], selective_prediction,
        np.clip(dt_p[covered_index], 1e-6, 1 - 1e-6),
    )
    summary_rows.append({
        "strategy": "engineering_agreement_rejector", "aurc": np.nan,
        "risk_at_engineering_coverage": engineering_risk,
        "coverage": engineering_coverage, "mcc": engineering_metrics["mcc"],
    })

    curve_table = pd.DataFrame(curve_rows)
    summary_table = pd.DataFrame(summary_rows)
    curve_table.to_csv(target, index=False)
    summary_table.to_csv(summary_target, index=False)

    style()
    fig, axis = plt.subplots(figsize=(3.45, 2.6), constrained_layout=True)
    colors = {"dt_softmax_response": "#0072B2", "xgb_softmax_response": "#009E73", "disagreement_graded": "#D55E00"}
    labels = {
        "dt_softmax_response": "DT softmax-response",
        "xgb_softmax_response": "XGB softmax-response",
        "disagreement_graded": "Graded disagreement",
    }
    for name in strategies:
        part = curve_table[curve_table["strategy"] == name].sort_values("coverage")
        axis.plot(part["coverage"], part["risk"], label=labels[name], color=colors[name], linewidth=1.1)
    axis.scatter(
        [engineering_coverage], [engineering_risk], color="black", marker="*", s=60, zorder=5,
        label="Engineering rejector",
    )
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk (1 - accuracy)")
    axis.legend(frameon=False, fontsize=6)
    axis.grid(color="0.88", linewidth=0.5)
    save(fig, output["figures"], "selective_risk_coverage_confirmation")

    manifest = {
        "status": "VALIDATED",
        "domain": "cic_ton_iot_confirmation", "seed": 42,
        "engineering_coverage": engineering_coverage, "engineering_risk": engineering_risk,
        "engineering_mcc": engineering_metrics["mcc"],
        "strategies": list(strategies.keys()),
        "limitations": [
            "Seed 42 only; the engineering rejector's own ten-seed variability "
            "(coverage 0.6937 +/- 0.0772) is not re-derived here.",
            "Softmax-response confidence is a standard but not necessarily optimal "
            "selective-classification rule; it establishes a reference frontier, "
            "not a claim that this is the best possible selective classifier.",
            "The base classifiers (DT, XGB probability branches) are unchanged from "
            "the frozen confirmation configuration; only the accept/reject rule varies.",
        ],
    }
    (output["manifests"] / "selective_classifier_baseline.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return curve_table, summary_table


if __name__ == "__main__":
    run()
