"""Task 1 (post-manuscript addendum): a second, independent external
confirmation-style evaluation on CIC-DDoS2019.

This module extends, but never imports-for-modification, the frozen
CIC-ToN-IoT confirmation protocol in benchmark/redesign_confirmation.py.
Every piece of the trained method is reused unmodified:

  - the same 68-feature canonical schema (benchmark.redesign_data.CANONICAL)
  - the same raw-to-canonical column mapping and validator
    (benchmark.redesign_confirmation.column_mapping) -- CIC-DDoS2019's
    Kaggle mirror (dhoogla/cicddos2019, the same curator already used for
    the CIC-ToN-IoT mirror) turned out to use the identical raw
    CICFlowMeter column-naming convention, so the mapping succeeds with
    zero new aliases
  - the same pooled-development-domain training sample
    (benchmark.redesign_selective.domain_balanced_sample over
    cicids2017 / cse_cic_ids2018 / cic_unsw_nb15)
  - the same tuned Decision Tree + XGBoost branches
    (benchmark.redesign_selective.tuned_pipeline, reading the same
    group_tuning_*.json hyperparameters already frozen for every other
    experiment in this project)
  - the same unlabeled, target-adaptive agreement rule
    (benchmark.redesign_selective.predict_selective)
  - the same 10 seeds (benchmark/redesign_config.yaml: study.seeds)

CIC-DDoS2019 labels are used ONLY to score already-frozen predictions.
They are never used to select, tune, or threshold anything -- the branch
models and thresholds are refit from the development domains only, exactly
as for CIC-ToN-IoT, and CIC-DDoS2019 is scored after the fact. This keeps
the comparison honest: any difference between this dataset's results and
the CIC-ToN-IoT confirmation results reflects the target domain, not a
different method.

The dataset (431,371 rows, 17 per-attack-type train/test parquet files
concatenated into one evaluation set) is small enough to hold in memory
directly; the checkpointed/streaming machinery in
redesign_confirmation.py (built for CIC-ToN-IoT's 4,847,499 rows) is not
needed and is not reused for that reason, not because a different
protocol was applied.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from benchmark.common import set_seed, write_json
from benchmark.metrics import binary_metrics
from benchmark.redesign_confirmation import column_mapping
from benchmark.redesign_data import CANONICAL
from benchmark.redesign_preprocessing import config, paths
from benchmark.redesign_selective import (
    DOMAIN_NAMES,
    domain_balanced_sample,
    predict_selective,
    tuned_pipeline,
)

RAW_DIR = Path("data/raw/cic_ddos2019_staging")


def _source_files():
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No CIC-DDoS2019 parquet files found under {RAW_DIR}")
    return files


def build_aligned_cache(force: bool = False) -> Path:
    output = paths()
    cache = output["data"] / "cic_ddos2019_second_confirmation_aligned"
    metadata_path = cache / "metadata.json"
    if metadata_path.exists() and not force:
        return cache
    cache.mkdir(parents=True, exist_ok=True)

    files = _source_files()
    combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    mapping = column_mapping(list(combined.columns))  # reused, unmodified
    feature_order = list(CANONICAL.values())
    features = combined[list(mapping)].rename(columns=mapping)[feature_order]
    features = features.apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)

    label_raw = combined.loc[valid, "Label"].astype(str)
    y = (label_raw.str.lower() != "benign").astype(np.int8).to_numpy()
    family_to_code = {name: i for i, name in enumerate(sorted(label_raw.unique()))}
    family = label_raw.map(family_to_code).astype(np.int16).to_numpy()
    X = features.loc[valid].clip(lower=0).to_numpy(dtype=np.float32)

    np.save(cache / "X.npy", X)
    np.save(cache / "y.npy", y)
    np.save(cache / "family.npy", family)
    metadata = {
        "source_files": [f.name for f in files],
        "raw_rows_total": int(len(combined)),
        "valid_rows": int(valid.sum()),
        "dropped_rows_non_finite": int((~valid).sum()),
        "feature_order": feature_order,
        "family_to_code": family_to_code,
        "positive_rate": float(y.mean()),
    }
    write_json(metadata_path, metadata)
    write_json(output["manifests"] / "cicddos2019_second_confirmation_dataset.json", {
        "dataset": "CIC-DDoS2019",
        "official_page": "https://www.unb.ca/cic/datasets/ddos-2019.html",
        "download_mirror": "https://www.kaggle.com/datasets/dhoogla/cicddos2019",
        "mirror_curator": (
            "dhoogla -- the same curator whose mirror is already used for "
            "the frozen CIC-ToN-IoT confirmation dataset"
        ),
        "archive_note": (
            "17 per-attack-type train/test parquet files, concatenated "
            "into one evaluation set; the original train/test split in "
            "the filenames is not used as a split here -- the whole "
            "dataset is one evaluation domain, mirroring how CIC-ToN-IoT "
            "was treated"
        ),
        "valid_aligned_rows": int(valid.sum()),
        "dropped_rows_non_finite": int((~valid).sum()),
        "features_aligned": len(feature_order),
        "column_mapping_source": (
            "benchmark.redesign_confirmation.column_mapping (imported, "
            "not duplicated or modified)"
        ),
        "labels_used_for_training_or_thresholds": False,
        "role": (
            "SECOND, independent external evaluation domain -- additive "
            "evidence only. Does NOT replace, modify, or touch the frozen "
            "CIC-ToN-IoT confirmation or its stored results."
        ),
    })
    return cache


def _oracle_threshold_mcc(y, probability, grid_points=197):
    y = np.asarray(y, dtype=int)
    best_mcc, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.01, 0.99, grid_points):
        mcc = matthews_corrcoef(y, (probability >= threshold).astype(int))
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, float(threshold)
    return best_mcc, best_threshold


def run(seeds: list[int] | None = None, force: bool = False):
    output = paths()
    cache = build_aligned_cache()
    metadata = json.loads((cache / "metadata.json").read_text())
    X = np.load(cache / "X.npy")
    y = np.load(cache / "y.npy")
    family = np.load(cache / "family.npy")
    family_names = {code: name for name, code in metadata["family_to_code"].items()}

    result_path = output["results"] / "cicddos2019_second_confirmation_ten_seed_runs.csv"
    rows = (
        [] if force or not result_path.exists()
        else pd.read_csv(result_path).to_dict("records")
    )
    completed = {int(r["seed"]) for r in rows}

    requested_seeds = seeds or config()["study"]["seeds"]
    for seed in requested_seeds:
        if seed in completed:
            continue
        X_train, y_train = domain_balanced_sample(DOMAIN_NAMES, seed)
        models, training_seconds = {}, 0.0
        for name in ("decision_tree", "xgboost"):
            set_seed(seed)
            model = tuned_pipeline(name, seed)
            started = time.perf_counter()
            model.fit(X_train, y_train)
            training_seconds += time.perf_counter() - started
            models[name] = model
        del X_train, y_train

        started = time.perf_counter()
        dt = models["decision_tree"].predict_proba(X)[:, 1].astype(np.float32)
        xgb = models["xgboost"].predict_proba(X)[:, 1].astype(np.float32)
        inference_seconds = time.perf_counter() - started
        selective = predict_selective(dt, xgb)
        mean_probability = ((dt.astype(np.float64) + xgb.astype(np.float64)) / 2).astype(np.float32)

        variants = (
            ("adaptive_decision_tree", np.ones(len(y), dtype=bool),
             selective["dt_prediction"], dt),
            ("selective_agreement_hybrid", selective["covered"],
             selective["dt_prediction"], mean_probability),
        )
        for variant, mask, prediction, probability in variants:
            metrics = binary_metrics(
                y[mask], prediction[mask].astype(int), probability[mask]
            )
            rows.append({
                "seed": seed, "variant": variant, "coverage": float(mask.mean()),
                "abstained": int((~mask).sum()),
                "abstained_attacks": int(((~mask) & (y == 1)).sum()),
                "dt_threshold": selective["dt_threshold"],
                "xgb_threshold": selective["xgb_threshold"],
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds, **metrics,
            })
        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(f"cicddos2019 second-confirmation seed={seed} complete", flush=True)

        if seed == 42:
            prediction_dir = output["results"] / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_dir / "cicddos2019_second_confirmation_seed42.npz",
                y=y, family=family,
                family_names=np.asarray(
                    [family_names[i] for i in sorted(family_names)], dtype="U64"
                ),
                dt_probability=dt, xgb_probability=xgb,
                covered=selective["covered"],
                selective_prediction=selective["selective_prediction"],
            )
            dt_oracle_mcc, dt_oracle_threshold = _oracle_threshold_mcc(y, dt)
            xgb_oracle_mcc, xgb_oracle_threshold = _oracle_threshold_mcc(y, xgb)
            write_json(output["tables"] / "cicddos2019_oracle_threshold_seed42.json", {
                "decision_tree": {
                    "roc_auc": float(roc_auc_score(y, dt)),
                    "frozen_threshold": float(selective["dt_threshold"]),
                    "fixed_frozen_mcc": float(matthews_corrcoef(
                        y, (dt >= selective["dt_threshold"]).astype(int)
                    )),
                    "oracle_threshold": dt_oracle_threshold,
                    "oracle_mcc": dt_oracle_mcc,
                    "mcc_recoverable_by_oracle_rethresholding": (
                        dt_oracle_mcc - float(matthews_corrcoef(
                            y, (dt >= selective["dt_threshold"]).astype(int)
                        ))
                    ),
                },
                "xgboost": {
                    "roc_auc": float(roc_auc_score(y, xgb)),
                    "frozen_threshold": float(selective["xgb_threshold"]),
                    "fixed_frozen_mcc": float(matthews_corrcoef(
                        y, (xgb >= selective["xgb_threshold"]).astype(int)
                    )),
                    "oracle_threshold": xgb_oracle_threshold,
                    "oracle_mcc": xgb_oracle_mcc,
                    "mcc_recoverable_by_oracle_rethresholding": (
                        xgb_oracle_mcc - float(matthews_corrcoef(
                            y, (xgb >= selective["xgb_threshold"]).astype(int)
                        ))
                    ),
                },
                "note": (
                    "Oracle values use CIC-DDoS2019 labels to pick the "
                    "threshold and are a non-deployable diagnostic upper "
                    "bound, not a proposed procedure. Seed 42 only, same "
                    "scope as the equivalent CIC-ToN-IoT diagnostic."
                ),
            })

    result = pd.DataFrame(rows)
    summary = result.groupby("variant")[
        ["coverage", "accuracy", "precision", "recall", "f1", "roc_auc",
         "pr_auc", "mcc", "tpr", "fpr", "ece", "brier",
         "negative_log_likelihood"]
    ].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(
        output["tables"] / "cicddos2019_second_confirmation_mean_std.csv", index=False
    )
    completed_seeds = sorted(result["seed"].unique().astype(int).tolist())
    expected_seeds = sorted(config()["study"]["seeds"])
    write_json(output["manifests"] / "cicddos2019_second_confirmation_execution.json", {
        "method_frozen": True,
        "labels_used_for_revision": False,
        "seeds_completed": completed_seeds,
        "seeds_expected": expected_seeds,
        "complete": completed_seeds == expected_seeds,
        "rows_evaluated": int(result["n_samples"].max()) if len(result) else 0,
        "status": (
            "SECOND_CONFIRMATION_COMPLETE"
            if completed_seeds == expected_seeds else "IN_PROGRESS"
        ),
    })
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    outcome = run(seeds=arguments.seed, force=arguments.force)
    print(outcome.groupby("variant")[["coverage", "mcc", "roc_auc", "tpr", "fpr"]].agg(["mean", "std"]))
