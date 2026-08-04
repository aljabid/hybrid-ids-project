"""Shift-mechanism and recalibration diagnostics.

Addresses two open items from the manuscript's REMAINING_WEAKNESSES.md
(maintained outside this repository):

  4. "Shift mechanism is unidentified." Marginal JSD (benchmark.redesign_analysis.drift)
     cannot separate covariate shift from label-prior shift. This module adds
     (a) explicit class-prior shift per domain, (b) label-conditional JSD
     (covariate shift computed separately within the benign and attack
     subpopulations), and (c) a ranking-vs-operating-point decomposition
     using ROC-AUC (prior-insensitive) against PR-AUC/fixed-threshold MCC
     (prior-sensitive), plus an oracle best-threshold MCC computed on each
     domain's own labels as a diagnostic upper bound (NOT a deployable
     procedure -- it uses target labels to pick the threshold and is
     reported only to show how much of the operating-point collapse is a
     thresholding problem versus a ranking problem).

  8. "Calibration analysis is descriptive... does not test post-shift
     recalibration methods using a predeclared unlabeled or labeled
     adaptation budget." This module fits temperature scaling (Guo et al.,
     already cited as [11]) and Platt scaling on a small stratified
     adaptation fold sampled from each domain's stored seed-42 predictions,
     then reports before/after ECE, Brier, NLL, and MCC on the disjoint
     evaluation fold.

Every analysis reuses only real, already-computed, already-checksummed
artifacts: benchmark.redesign_analysis._jsd for feature drift, and the
seed-42 raw prediction archives in results/predictions/. No model is
retrained. Scope is explicitly seed-42 only, consistent with the existing
reliability-panel and focused-error-analysis precedent in
benchmark/final_publication_analysis.py.
"""

from __future__ import annotations

import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from benchmark.metrics import binary_metrics, expected_calibration_error
from benchmark.redesign_analysis import _jsd
from benchmark.redesign_preprocessing import load_partition, paths

MODELS = ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
LABELS = {
    "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
}
DEV_DOMAINS = ["cse_cic_ids2018", "cic_unsw_nb15"]
DOMAIN_LABELS = {
    "cse_cic_ids2018": "CSE-CIC-IDS2018", "cic_unsw_nb15": "CIC-UNSW-NB15",
    "cic_ton_iot_confirmation": "CIC-ToN-IoT confirmation",
}


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


# ---------------------------------------------------------------------------
# 1. Class-prior shift and label-conditional feature drift
# ---------------------------------------------------------------------------

def _domain_xy(output, name, rng, cap=100_000):
    if name == "cic_ton_iot_confirmation":
        metadata = json.loads(
            (output["data"] / "cic_ton_iot_confirmation_aligned/metadata.json").read_text()
        )
        n = metadata["valid_rows"]
        X = np.load(output["data"] / "cic_ton_iot_confirmation_aligned/X.npy", mmap_mode="r")[:n]
        y = np.load(output["data"] / "cic_ton_iot_confirmation_aligned/y.npy", mmap_mode="r")[:n]
    else:
        with np.load(output["data"] / f"{name}_expanded_diagnostic.npz") as data:
            X, y = np.asarray(data["X"]), np.asarray(data["y"])
    if len(X) > cap:
        index = rng.choice(len(X), cap, replace=False)
        X, y = X[index], y[index]
    return X, y


def prior_and_label_conditional_drift(force=False):
    output = paths()
    prior_target = output["tables"] / "prior_shift_summary.csv"
    drift_target = output["tables"] / "label_conditional_feature_drift.csv"
    if prior_target.exists() and drift_target.exists() and not force:
        return pd.read_csv(prior_target), pd.read_csv(drift_target)

    rng = np.random.default_rng(42)
    Xtr, ytr, _, _, features = load_partition("train")
    train_index = rng.choice(len(Xtr), min(100_000, len(Xtr)), replace=False)
    Xtr, ytr = Xtr[train_index], ytr[train_index]
    train_prior = float(np.mean(ytr))

    prior_rows = [{
        "domain": "cicids2017_train", "attack_prior": train_prior,
        "prior_odds_ratio_vs_train": 1.0, "rows": len(ytr),
    }]
    drift_rows = []
    for domain in DEV_DOMAINS + ["cic_ton_iot_confirmation"]:
        Xd, yd = _domain_xy(output, domain, rng)
        domain_prior = float(np.mean(yd))
        train_odds = train_prior / (1 - train_prior)
        domain_odds = domain_prior / (1 - domain_prior)
        prior_rows.append({
            "domain": domain, "attack_prior": domain_prior,
            "prior_odds_ratio_vs_train": domain_odds / train_odds, "rows": len(yd),
        })
        for label, name in ((0, "benign"), (1, "attack")):
            base_mask, target_mask = ytr == label, yd == label
            for index, feature in enumerate(features):
                drift_rows.append({
                    "domain": domain, "label_condition": name, "feature": feature,
                    "js_divergence": _jsd(Xtr[base_mask, index], Xd[target_mask, index]),
                    "base_n": int(base_mask.sum()), "target_n": int(target_mask.sum()),
                })
        print(f"label-conditional drift computed for {domain}", flush=True)

    prior_table = pd.DataFrame(prior_rows)
    drift_table = pd.DataFrame(drift_rows)
    prior_table.to_csv(prior_target, index=False)
    drift_table.to_csv(drift_target, index=False)

    style()
    summary = drift_table.groupby(["domain", "label_condition"])["js_divergence"].mean().reset_index()
    marginal = pd.read_csv(output["tables"] / "feature_drift_jsd.csv").groupby("dataset")["js_divergence"].mean()
    domains = DEV_DOMAINS + ["cic_ton_iot_confirmation"]
    x = np.arange(len(domains))
    width = 0.25
    fig, axis = plt.subplots(figsize=(3.45, 2.6), constrained_layout=True)
    for offset, (condition, color) in zip(
        (-width, 0, width),
        (("benign", "#0072B2"), ("attack", "#D55E00"), ("marginal", "#009E73")),
    ):
        if condition == "marginal":
            values = [marginal.get(d, np.nan) for d in domains]
        else:
            values = [
                summary[(summary["domain"] == d) & (summary["label_condition"] == condition)]["js_divergence"].mean()
                for d in domains
            ]
        axis.bar(x + offset, values, width, label=condition, color=color, edgecolor="black", linewidth=0.4)
    axis.set_xticks(x)
    axis.set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=15, ha="right")
    axis.set_ylabel("Mean feature JSD")
    axis.legend(frameon=False, fontsize=6)
    axis.grid(axis="y", color="0.88", linewidth=0.5)
    save(fig, output["figures"], "label_conditional_drift")
    return prior_table, drift_table


# ---------------------------------------------------------------------------
# 2. Ranking-vs-operating-point decomposition (oracle threshold, seed 42)
# ---------------------------------------------------------------------------

def _load_seed42(output, model, domain):
    path = output["results"] / "predictions" / f"{model}_{domain}_seed_42.npz"
    with np.load(path) as data:
        return data["y_true"], data["probability"]


def _oracle_threshold_mcc(y, probability):
    # Only MCC (via the confusion-matrix counts) varies with threshold; ROC-AUC,
    # PR-AUC, ECE, Brier, and NLL do not depend on the swept decision rule, so
    # calling the full binary_metrics() per grid point would recompute those
    # unnecessarily (and expensively) 197 times. matthews_corrcoef alone is
    # O(n) per threshold and is all this diagnostic upper-bound search needs.
    y = np.asarray(y, dtype=int)
    best_mcc, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.01, 0.99, 197):
        mcc = matthews_corrcoef(y, (probability >= threshold).astype(int))
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, float(threshold)
    return best_mcc, best_threshold


def threshold_sensitivity(force=False):
    output = paths()
    target = output["tables"] / "threshold_sensitivity_seed42.csv"
    if target.exists() and not force:
        return pd.read_csv(target)
    rows = []
    for domain in DEV_DOMAINS:
        for model in MODELS:
            y, probability = _load_seed42(output, model, domain)
            fixed = binary_metrics(y, probability >= 0.5, probability)
            oracle_mcc, oracle_threshold = _oracle_threshold_mcc(y, probability)
            rows.append({
                "domain": domain, "model": model,
                "roc_auc": float(roc_auc_score(y, probability)),
                "pr_auc": fixed["pr_auc"],
                "fixed_threshold_mcc": fixed["mcc"], "fixed_threshold_tpr": fixed["tpr"],
                "fixed_threshold_fpr": fixed["fpr"],
                "oracle_threshold": oracle_threshold, "oracle_threshold_mcc": oracle_mcc,
                "mcc_recoverable_by_rethresholding": oracle_mcc - fixed["mcc"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(target, index=False)
    return table


# ---------------------------------------------------------------------------
# 3. Recalibration experiment: temperature scaling + Platt scaling
# ---------------------------------------------------------------------------

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_temperature(y, probability, grid=None):
    grid = grid if grid is not None else np.geomspace(0.02, 500.0, 800)
    z = _logit(probability)
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = np.clip(_sigmoid(z / t), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    if best_t >= 0.95 * grid[-1]:
        raise RuntimeError(
            f"Temperature grid upper bound reached (T={best_t:.3f}); widen the grid."
        )
    return best_t


def _apply_temperature(probability, t):
    return _sigmoid(_logit(probability) / t)


def _fit_platt(y, probability, seed=42):
    z = _logit(probability).reshape(-1, 1)
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(z, y)
    return model


def _apply_platt(model, probability):
    z = _logit(probability).reshape(-1, 1)
    return model.predict_proba(z)[:, 1]


def _calibration_row(y, probability):
    ece, _ = expected_calibration_error(y, probability)
    metrics = binary_metrics(y, probability >= 0.5, probability)
    return {
        "ece": ece, "brier": metrics["brier"],
        "nll": metrics["negative_log_likelihood"], "mcc": metrics["mcc"],
        "tpr": metrics["tpr"], "fpr": metrics["fpr"],
    }


def recalibration_experiment(seed=42, adaptation_rows=2000, force=False):
    output = paths()
    target = output["tables"] / "recalibration_seed42.csv"
    if target.exists() and not force:
        return pd.read_csv(target)
    rng = np.random.default_rng(seed)
    rows = []
    for domain in DEV_DOMAINS:
        for model in MODELS:
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

            baseline = _calibration_row(y_eval, p_eval)
            temperature = _fit_temperature(y_adapt, p_adapt)
            p_temp = _apply_temperature(p_eval, temperature)
            temp_row = _calibration_row(y_eval, p_temp)
            platt_model = _fit_platt(y_adapt, p_adapt, seed)
            p_platt = _apply_platt(platt_model, p_eval)
            platt_row = _calibration_row(y_eval, p_platt)

            rows.append({
                "domain": domain, "model": model,
                "adaptation_rows": len(adapt_idx), "evaluation_rows": len(eval_idx),
                "temperature": temperature,
                **{f"baseline_{k}": v for k, v in baseline.items()},
                **{f"temperature_scaled_{k}": v for k, v in temp_row.items()},
                **{f"platt_scaled_{k}": v for k, v in platt_row.items()},
                "ece_reduction_temperature": baseline["ece"] - temp_row["ece"],
                "ece_reduction_platt": baseline["ece"] - platt_row["ece"],
            })
            print(f"recalibration {model}/{domain}: "
                  f"ECE {baseline['ece']:.4f} -> T={temp_row['ece']:.4f} "
                  f"Platt={platt_row['ece']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(target, index=False)

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), constrained_layout=True, sharey=True)
    width = 0.27
    for panel_index, (axis, domain) in enumerate(zip(axes, DEV_DOMAINS)):
        part = table[table["domain"] == domain].set_index("model").loc[MODELS].reset_index()
        x = np.arange(len(part))
        axis.bar(x - width, part["baseline_ece"], width, label="Uncalibrated", color="#999999", edgecolor="black", linewidth=0.4)
        axis.bar(x, part["temperature_scaled_ece"], width, label="Temperature-scaled", color="#0072B2", edgecolor="black", linewidth=0.4)
        axis.bar(x + width, part["platt_scaled_ece"], width, label="Platt-scaled", color="#E69F00", edgecolor="black", linewidth=0.4)
        axis.set_xticks(x)
        axis.set_xticklabels([LABELS[m] for m in part["model"]], rotation=20, ha="right")
        axis.set_title(DOMAIN_LABELS[domain])
        axis.grid(axis="y", color="0.88", linewidth=0.5)
        if panel_index > 0:
            axis.tick_params(labelleft=False)
    axes[0].set_ylabel("ECE (held-out evaluation fold)")
    axes[0].legend(frameon=False, ncol=1, loc="upper left", fontsize=6)
    save(fig, output["figures"], "recalibration_ece_comparison")
    return table


def main():
    prior, drift = prior_and_label_conditional_drift()
    sensitivity = threshold_sensitivity()
    recalibration = recalibration_experiment()
    manifest = {
        "status": "VALIDATED",
        "tasks": [
            "class_prior_shift", "label_conditional_feature_drift",
            "roc_auc_vs_pr_auc_threshold_sensitivity_oracle_upper_bound",
            "temperature_and_platt_recalibration_seed42",
        ],
        "prior_rows": len(prior), "drift_rows": len(drift),
        "sensitivity_rows": len(sensitivity), "recalibration_rows": len(recalibration),
        "adaptation_budget_rows": 2000,
        "limitations": [
            "Label-conditional JSD and prior-shift use CICIDS2017 training vs each "
            "development/confirmation domain; it separates covariate-given-label shift "
            "from prior shift but cannot further separate concept shift from extractor shift.",
            "Oracle-threshold MCC uses each domain's own labels to select the threshold "
            "and is a diagnostic upper bound on what rethresholding alone could achieve, "
            "not an unbiased or deployable estimate.",
            "Recalibration is evaluated at seed 42 only, using stored raw predictions; "
            "the adaptation fold (2,000 flows) is a hypothetical small labeled budget, "
            "not validated across the ten-seed protocol.",
            "Temperature scaling is monotonic and therefore leaves 0.5-threshold MCC/TPR/FPR "
            "unchanged by construction; only ECE/Brier/NLL can improve. Platt scaling can "
            "additionally shift the decision boundary and is reported for comparison.",
        ],
    }
    output = paths()
    (output["manifests"] / "shift_and_calibration_diagnostics.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
