                                                                                  

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from benchmark.common import set_seed, write_json
from benchmark.metrics import binary_metrics
from benchmark.redesign_preprocessing import config, load_partition, paths
from benchmark.redesign_supervised import model_and_space, selected_transformer


DOMAIN_NAMES = ("cicids2017", "cse_cic_ids2018", "cic_unsw_nb15")


def sample_domain(name, seed, per_class=50000):
    rng = np.random.default_rng(seed)
    if name == "cicids2017":
        X_parts, y_parts = [], []
        for offset, partition in enumerate(("train", "validation", "internal_test")):
            X, y, _, _, _ = load_partition(partition)
            for label in (0, 1):
                candidates = np.flatnonzero(y == label)
                take = min(per_class, len(candidates))
                selected = np.random.default_rng(seed + offset).choice(
                    candidates, take, replace=False
                )
                X_parts.append(X[selected]); y_parts.append(y[selected])
            del X, y
        X_pool, y_pool = np.concatenate(X_parts), np.concatenate(y_parts)
        index = np.concatenate([
            rng.choice(
                np.flatnonzero(y_pool == label),
                min(per_class, int((y_pool == label).sum())),
                replace=False,
            )
            for label in (0, 1)
        ])
        X, y = X_pool[index], y_pool[index]
    else:
        with np.load(
            paths()["data"] / f"{name}_expanded_diagnostic.npz",
            allow_pickle=False,
        ) as data:
            y_all = data["y"]
            index = np.concatenate([
                rng.choice(
                    np.flatnonzero(y_all == label),
                    min(per_class, int((y_all == label).sum())),
                    replace=False,
                )
                for label in (0, 1)
            ])
            X, y = data["X"][index], y_all[index]
    order = rng.permutation(len(y))
    return X[order], y[order]


def domain_balanced_sample(names, seed, per_class=50000):
    samples = [sample_domain(name, seed, per_class) for name in names]
    X = np.concatenate([sample[0] for sample in samples])
    y = np.concatenate([sample[1] for sample in samples])
    order = np.random.default_rng(seed).permutation(len(y))
    return X[order], y[order]


def score_domain(name, models):
    y_parts, family_parts, dt_parts, xgb_parts = [], [], [], []
    if name == "cicids2017":
        def sources():
            for partition in ("train", "validation", "internal_test"):
                X, y, family, _, _ = load_partition(partition)
                yield X, y, family
    else:
        def sources():
            data = np.load(
                paths()["data"] / f"{name}_expanded_diagnostic.npz",
                allow_pickle=False,
            )
            yield data["X"], data["y"], data["family"]
    for X, y, family in sources():
        y_parts.append(y.copy()); family_parts.append(family.copy())
        dt_parts.append(models["decision_tree"].predict_proba(X)[:, 1])
        xgb_parts.append(models["xgboost"].predict_proba(X)[:, 1])
        del X, y, family
    return (
        np.concatenate(y_parts), np.concatenate(family_parts),
        np.concatenate(dt_parts), np.concatenate(xgb_parts),
    )


def adaptive_thresholds(dt_probability, xgb_probability):
    sample_index = np.linspace(
        0, len(dt_probability) - 1, min(30000, len(dt_probability)), dtype=int
    )
    sample = np.clip(dt_probability[sample_index], 1e-6, 1 - 1e-6)
    logit = np.log(sample / (1 - sample)).reshape(-1, 1)
    try:
        mixture = GaussianMixture(
            2, random_state=42, n_init=1, reg_covar=1e-5
        ).fit(logit)
        high = int(np.argmax(mixture.means_.ravel()))
        grid = np.linspace(
            np.quantile(logit, 0.001), np.quantile(logit, 0.999), 1000
        ).reshape(-1, 1)
        posterior = mixture.predict_proba(grid)[:, high]
        logit_threshold = float(grid[np.argmin(np.abs(posterior - 0.5)), 0])
        dt_threshold = float(1 / (1 + np.exp(-logit_threshold)))
    except Exception:
        dt_threshold = 0.5
    xgb_threshold = float(
        np.median(xgb_probability)
        + 6.0 * (
            np.quantile(xgb_probability, 0.75)
            - np.quantile(xgb_probability, 0.25)
        )
    )
    return dt_threshold, xgb_threshold


def predict_selective(dt_probability, xgb_probability):
    dt_threshold, xgb_threshold = adaptive_thresholds(
        dt_probability, xgb_probability
    )
    dt_prediction = dt_probability >= dt_threshold
    xgb_prediction = xgb_probability >= xgb_threshold
    covered = dt_prediction == xgb_prediction
    return {
        "dt_threshold": dt_threshold, "xgb_threshold": xgb_threshold,
        "dt_prediction": dt_prediction, "xgb_prediction": xgb_prediction,
        "covered": covered, "selective_prediction": dt_prediction[covered],
    }


def tuned_pipeline(name, seed):
    output = paths()
    tuning = json.loads(
        (output["results"] / f"group_tuning_{name}.json").read_text()
    )
    estimator, _ = model_and_space(name, seed)
    pipeline = Pipeline([
        ("preprocessor", selected_transformer(seed)), ("model", estimator)
    ])
    pipeline.set_params(**tuning["best_parameters"])
    return pipeline


def validate():
    output = paths()
    checkpoint = output["results"] / "leave_domain_out_selective_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(int(row["seed"]), row["held_out_domain"]) for row in rows}
    for seed in config()["study"]["seeds"]:
        for held_out in DOMAIN_NAMES:
            if (seed, held_out) in completed:
                continue
            training_domains = [name for name in DOMAIN_NAMES if name != held_out]
            X_train, y_train = domain_balanced_sample(
                training_domains, seed
            )
            models = {}
            training_seconds = 0.0
            for name in ("decision_tree", "xgboost"):
                set_seed(seed)
                model = tuned_pipeline(name, seed)
                started = time.perf_counter()
                model.fit(X_train, y_train)
                training_seconds += time.perf_counter() - started
                models[name] = model
            y_test, _, dt, xgb = score_domain(held_out, models)
            selective = predict_selective(dt, xgb)
            for variant, mask, prediction, probability in (
                (
                    "adaptive_decision_tree", np.ones(len(y_test), dtype=bool),
                    selective["dt_prediction"], dt,
                ),
                (
                    "selective_agreement_hybrid", selective["covered"],
                    selective["selective_prediction"],
                    (dt[selective["covered"]] + xgb[selective["covered"]]) / 2,
                ),
            ):
                metrics = binary_metrics(
                    y_test[mask], prediction, probability
                )
                rows.append({
                    "seed": seed, "held_out_domain": held_out,
                    "variant": variant, "coverage": float(mask.mean()),
                    "abstained": int((~mask).sum()),
                    "abstained_attacks": int(((~mask) & (y_test == 1)).sum()),
                    "dt_threshold": selective["dt_threshold"],
                    "xgb_threshold": selective["xgb_threshold"],
                    "training_seconds": training_seconds, **metrics,
                })
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"leave-domain seed={seed} held={held_out}", flush=True)
    result = pd.DataFrame(rows)
    summary = result.groupby(["held_out_domain", "variant"])[
        ["coverage", "mcc", "pr_auc", "tpr", "fpr"]
    ].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(
        output["tables"] / "leave_domain_out_selective_mean_std.csv", index=False
    )
    return result


def freeze_final_method():
    output = paths()
    results = pd.read_csv(
        output["results"] / "leave_domain_out_selective_runs.csv"
    )
    selective = results[results["variant"] == "selective_agreement_hybrid"]
    payload = {
        "status": "FINAL_METHOD_FROZEN_BEFORE_CONFIRMATION_DOWNLOAD",
        "training_domains": [
            "CICIDS2017", "CSE-CIC-IDS2018 diagnostic",
            "CIC-UNSW-NB15 diagnostic",
        ],
        "training_sampling": "per-domain, per-binary-class cap of 50,000",
        "features": "68-feature semantic CICFlowMeter schema v3",
        "preprocessing": "winsorized log14 RobustScaler learned on pooled training only",
        "branches": ["tuned Decision Tree", "tuned XGBoost"],
        "decision_tree_rule": "two-component Gaussian mixture in logit score space",
        "xgboost_rule": "median + 6 * IQR of unlabeled target scores",
        "output_rule": "predict only when branch decisions agree; otherwise abstain",
        "development_mean_coverage": float(selective["coverage"].mean()),
        "development_worst_domain_mean_mcc": float(
            selective.groupby("held_out_domain")["mcc"].mean().min()
        ),
        "development_worst_domain_mean_fpr": float(
            selective.groupby("held_out_domain")["fpr"].mean().max()
        ),
        "confirmation_dataset": "CIC-ToN-IoT",
        "confirmation_source": (
            "https://staff.itee.uq.edu.au/marius/NIDS_datasets/ "
            "(official UQ ML-Based NIDS repository)"
        ),
        "confirmation_downloaded": False,
        "confirmation_labels_used_for_revision": False,
        "seeds": config()["study"]["seeds"],
    }
    write_json(output["manifests"] / "final_method_freeze.json", payload)
    return payload


if __name__ == "__main__":
    validate()
    print(json.dumps(freeze_final_method(), indent=2))
