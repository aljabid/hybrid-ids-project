"""Robustness/uncertainty-quantification extension (post-manuscript addendum).

Answers the "how robust are these numbers" question for the four headline
seed-level experiments WITHOUT changing the frozen protocol, retraining any
model, or touching the untouched confirmation dataset's decision procedure.
Two, independent techniques are applied, both reusing already-computed,
already-checksummed artifacts only:

  1. Seed-level bootstrap 95% CIs (`extended_bootstrap_intervals`) --
     the exact same resampling methodology as
     `benchmark.redesign_analysis.bootstrap_intervals` (rng seed 42,
     20,000 repetitions, resampling the 10 per-seed metric values with
     replacement), extended to three experiments it did not originally
     cover: the matched split-strategy control (Table II), the frozen
     confirmation domain (Table VI), and the held-out-family study
     (Table IV). This directly quantifies how much the headline MCC
     figures would plausibly move under a different draw of the same
     10-seed protocol.

  2. Calibration-sensitivity sweeps (`temperature_sensitivity`,
     `adaptation_fold_sensitivity`) -- using the stored seed-42
     predictions and the exact adapt/eval split logic already used by
     `benchmark.shift_and_calibration_diagnostics.recalibration_experiment`,
     quantify how much the recalibration benefit depends on (a) hitting
     the fitted temperature exactly and (b) the size of the adaptation
     fold. No model is retrained; only the already-frozen probability
     outputs are re-scaled.

Every number written here is traceable to the seed-level CSVs and
seed-42 .npz prediction archives already present in
artifacts/redesign_v3/results/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.common import write_json
from benchmark.metrics import binary_metrics
from benchmark.redesign_preprocessing import paths
from benchmark.shift_and_calibration_diagnostics import (
    DEV_DOMAINS,
    MODELS as DIAG_MODELS,
    _apply_temperature,
    _calibration_row,
    _fit_temperature,
    _load_seed42,
)

METRICS_FULL = ("mcc", "pr_auc", "tpr", "fpr", "ece", "brier")
METRICS_MATCHED = ("mcc", "tpr", "fpr", "pr_auc", "ece")


def _bootstrap_group(part: pd.DataFrame, metrics, rng, repetitions=20_000):
    rows = []
    for metric in metrics:
        values = part.sort_values("seed")[metric].to_numpy()
        samples = values[rng.integers(0, len(values), (repetitions, len(values)))]
        means = samples.mean(axis=1)
        rows.append({
            "metric": metric,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)),
            "n_seeds": int(len(values)),
            "bootstrap_repetitions": repetitions,
        })
    return rows


def extended_bootstrap_intervals(force=False):
    """Seed-level bootstrap CIs for Table II, Table IV, and Table VI."""
    output = paths()
    target = output["tables"] / "robustness_bootstrap_extended.csv"
    if target.exists() and not force:
        return pd.read_csv(target)

    specs = [
        (
            "matched_split_table_ii",
            output["tables"] / "matched_split_combined_metrics.csv",
            ["split_strategy", "model"], METRICS_MATCHED,
        ),
        (
            "held_out_family_table_iv",
            output["results"] / "unknown_family_ten_seed_runs.csv",
            ["held_out_family", "variant"], METRICS_FULL,
        ),
        (
            "frozen_confirmation_table_vi",
            output["results"] / "confirmation_ten_seed_runs.csv",
            ["variant"], METRICS_FULL,
        ),
    ]

    all_rows = []
    for experiment, source, group_cols, metrics in specs:
        data = pd.read_csv(source)
        rng = np.random.default_rng(42)  # same seed as the original bootstrap_intervals
        for key, part in data.groupby(group_cols):
            keys = key if isinstance(key, tuple) else (key,)
            for row in _bootstrap_group(part, metrics, rng):
                row["experiment"] = experiment
                row.update(dict(zip(group_cols, keys)))
                all_rows.append(row)

    table = pd.DataFrame(all_rows)
    ordered_cols = ["experiment"] + [
        c for c in table.columns if c not in ("experiment", "metric", "mean", "std",
                                               "ci95_low", "ci95_high", "n_seeds",
                                               "bootstrap_repetitions")
    ] + ["metric", "mean", "std", "ci95_low", "ci95_high", "n_seeds", "bootstrap_repetitions"]
    table = table[ordered_cols]
    table.to_csv(target, index=False)
    return table


def temperature_sensitivity(seed=42, adaptation_rows=2000, force=False):
    """Sweep the recalibration temperature around its fitted optimum.

    Reuses the identical adapt/eval split as recalibration_experiment (same
    rng seed, same adaptation_rows) so the fitted T* matches
    recalibration_seed42.csv exactly; only the applied multiplier varies.
    """
    output = paths()
    target = output["tables"] / "temperature_sensitivity_seed42.csv"
    if target.exists() and not force:
        return pd.read_csv(target)

    multipliers = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    rng = np.random.default_rng(seed)
    rows = []
    for domain in DEV_DOMAINS:
        for model in DIAG_MODELS:
            y, probability = _load_seed42(output, model, domain)
            n_per_class = adaptation_rows // 2
            adapt_idx = []
            for label in (0, 1):
                pool = np.flatnonzero(y == label)
                take = min(n_per_class, len(pool))
                adapt_idx.append(rng.choice(pool, take, replace=False))
            adapt_idx = np.concatenate(adapt_idx)
            eval_idx = np.setdiff1d(np.arange(len(y)), adapt_idx, assume_unique=False)
            y_adapt, p_adapt = y[adapt_idx], probability[adapt_idx]
            y_eval, p_eval = y[eval_idx], probability[eval_idx]

            t_star = _fit_temperature(y_adapt, p_adapt)
            for multiplier in multipliers:
                t = t_star * multiplier
                p_scaled = _apply_temperature(p_eval, t)
                metrics = _calibration_row(y_eval, p_scaled)
                rows.append({
                    "domain": domain, "model": model,
                    "fitted_temperature": t_star, "multiplier": multiplier,
                    "applied_temperature": t,
                    **metrics,
                })
            print(f"temperature sensitivity {model}/{domain}: T*={t_star:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(target, index=False)
    return table


def adaptation_fold_sensitivity(fold_sizes=(500, 1000, 2000, 4000), n_repeats=5, force=False):
    """Refit temperature on differently sized/sampled adaptation folds.

    For each fold size, draw n_repeats independent stratified adaptation
    folds (different rng seeds) from the same seed-42 predictions, refit T,
    and evaluate on the disjoint remainder. Reports mean +/- SD of the
    fitted temperature and resulting ECE/MCC across repeats, quantifying
    sensitivity to which rows land in the adaptation split.
    """
    output = paths()
    target = output["tables"] / "adaptation_fold_sensitivity_seed42.csv"
    if target.exists() and not force:
        return pd.read_csv(target)

    rows = []
    failures = []
    # Widened beyond the 500 upper bound used elsewhere in this project:
    # small adaptation folds are expected to occasionally produce extreme
    # or non-convergent fits, and that instability is itself part of what
    # this sensitivity sweep is measuring, not a boundary to hide.
    wide_grid = np.geomspace(0.02, 5000.0, 1000)
    for domain in DEV_DOMAINS:
        for model in DIAG_MODELS:
            y, probability = _load_seed42(output, model, domain)
            for fold_size in fold_sizes:
                for repeat in range(n_repeats):
                    rng = np.random.default_rng(1000 * fold_size + repeat)
                    n_per_class = fold_size // 2
                    adapt_idx = []
                    for label in (0, 1):
                        pool = np.flatnonzero(y == label)
                        take = min(n_per_class, len(pool))
                        adapt_idx.append(rng.choice(pool, take, replace=False))
                    adapt_idx = np.concatenate(adapt_idx)
                    eval_idx = np.setdiff1d(np.arange(len(y)), adapt_idx, assume_unique=False)
                    y_adapt, p_adapt = y[adapt_idx], probability[adapt_idx]
                    y_eval, p_eval = y[eval_idx], probability[eval_idx]

                    try:
                        t = _fit_temperature(y_adapt, p_adapt, grid=wide_grid)
                    except RuntimeError:
                        failures.append({
                            "domain": domain, "model": model,
                            "adaptation_fold_size": fold_size, "repeat": repeat,
                        })
                        continue
                    p_scaled = _apply_temperature(p_eval, t)
                    metrics = _calibration_row(y_eval, p_scaled)
                    rows.append({
                        "domain": domain, "model": model,
                        "adaptation_fold_size": fold_size, "repeat": repeat,
                        "fitted_temperature": t,
                        **metrics,
                    })
            print(f"adaptation-fold sensitivity done: {model}/{domain} "
                  f"({len(failures)} non-convergent fits so far)", flush=True)
    per_row = pd.DataFrame(rows)
    per_row.to_csv(output["tables"] / "adaptation_fold_sensitivity_seed42_raw.csv", index=False)
    pd.DataFrame(failures).to_csv(
        output["tables"] / "adaptation_fold_sensitivity_seed42_nonconvergent.csv", index=False
    )

    summary = per_row.groupby(["domain", "model", "adaptation_fold_size"]).agg(
        fitted_temperature_mean=("fitted_temperature", "mean"),
        fitted_temperature_std=("fitted_temperature", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
    ).reset_index()
    fail_counts = (
        pd.DataFrame(failures).groupby(["domain", "model", "adaptation_fold_size"]).size()
        if failures else pd.Series(dtype=int)
    )
    summary["n_repeats_attempted"] = n_repeats
    summary["n_nonconvergent"] = summary.apply(
        lambda r: int(fail_counts.get((r["domain"], r["model"], r["adaptation_fold_size"]), 0)),
        axis=1,
    )
    summary.to_csv(target, index=False)
    return summary


def run_all():
    print("=== extended bootstrap intervals (Table II / IV / VI) ===", flush=True)
    bootstrap = extended_bootstrap_intervals(force=True)
    print("=== temperature sensitivity sweep ===", flush=True)
    temp_sens = temperature_sensitivity(force=True)
    print("=== adaptation-fold sensitivity ===", flush=True)
    fold_sens = adaptation_fold_sensitivity(force=True)

    output = paths()
    write_json(output["manifests"] / "robustness_extension_manifest.json", {
        "description": (
            "Post-manuscript robustness addendum: seed-level bootstrap CIs "
            "extended to Table II/IV/VI, plus calibration-temperature and "
            "adaptation-fold sensitivity sweeps. No model retrained; no "
            "frozen confirmation procedure modified; all inputs are the "
            "already-committed seed-level CSVs and seed-42 .npz prediction "
            "archives under artifacts/redesign_v3/results/."
        ),
        "outputs": {
            "robustness_bootstrap_extended.csv": {
                "rows": len(bootstrap), "method": "seed-level bootstrap, "
                "rng=default_rng(42), 20000 repetitions per group",
            },
            "temperature_sensitivity_seed42.csv": {
                "rows": len(temp_sens),
                "multipliers": [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0],
            },
            "adaptation_fold_sensitivity_seed42.csv": {
                "rows": len(fold_sens),
                "fold_sizes": [500, 1000, 2000, 4000], "repeats_per_size": 5,
            },
        },
    })
    print("Done. See artifacts/redesign_v3/tables/ and manifests/robustness_extension_manifest.json")


if __name__ == "__main__":
    run_all()
