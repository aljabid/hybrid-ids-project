                                                                     

                                                                           
                                                                         
                                                                 
   

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline

from benchmark.common import set_seed, write_json
from benchmark.redesign_data import CANONICAL
from benchmark.redesign_preprocessing import config, paths
from benchmark.redesign_selective import (
    DOMAIN_NAMES,
    domain_balanced_sample,
    predict_selective,
    tuned_pipeline,
)
from benchmark.redesign_supervised import model_and_space, selected_transformer


RAW = Path("data/raw/cic_ton_iot_confirmation/CIC-ToN-IoT-V2.parquet")
ARCHIVE = Path("data/raw/cic_ton_iot_confirmation/cictoniot_v2.zip")
ARCHIVE_SHA256 = "c434fca8f461073ebfcf127df5889211ee1b29fba9bf72444091bf0707b4ec2c"

                                                                             
ALIASES = {
    "Fwd Packets Length Total": "Total Length of Fwd Packets",
    "Bwd Packets Length Total": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "Avg Packet Size": "Average Packet Size",
    "Fwd Act Data Packets": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def column_mapping(schema_names: list[str]) -> dict[str, str]:
    mapping = {}
    for source in schema_names:
        canonical_source = ALIASES.get(source, source)
        if canonical_source in CANONICAL:
            mapping[source] = CANONICAL[canonical_source]
    expected = set(CANONICAL.values())
    if set(mapping.values()) != expected:
        raise ValueError(
            "Confirmation feature alignment failed; missing "
            f"{sorted(expected - set(mapping.values()))}"
        )
    return mapping


def build_aligned_cache(force: bool = False) -> Path:
    output = paths()
    cache = output["data"] / "cic_ton_iot_confirmation_aligned"
    metadata_path = cache / "metadata.json"
    if metadata_path.exists() and not force:
        return cache
    if not RAW.exists() or sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise RuntimeError("Confirmation dataset missing or archive checksum mismatch")

    parquet = pq.ParquetFile(RAW)
    mapping = column_mapping(parquet.schema_arrow.names)
    source_columns = list(mapping) + ["Label", "Attack"]
    feature_order = list(CANONICAL.values())
    cache.mkdir(parents=True, exist_ok=True)
    total_rows = parquet.metadata.num_rows
    X_store = np.lib.format.open_memmap(
        cache / "X.npy", mode="w+", dtype=np.float32,
        shape=(total_rows, len(feature_order)),
    )
    y_store = np.lib.format.open_memmap(
        cache / "y.npy", mode="w+", dtype=np.int8, shape=(total_rows,),
    )
    family_store = np.lib.format.open_memmap(
        cache / "family.npy", mode="w+", dtype=np.int16, shape=(total_rows,),
    )
    family_to_code: dict[str, int] = {}
    written = 0
    for batch in parquet.iter_batches(batch_size=100_000, columns=source_columns):
        frame = batch.to_pandas()
        features = frame[list(mapping)].rename(columns=mapping)[feature_order]
        features = features.apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
        X_batch = features.loc[valid].clip(lower=0).to_numpy(dtype=np.float32)
        y_batch = (
            pd.to_numeric(frame.loc[valid, "Label"], errors="coerce")
            .fillna(0).astype(np.int8).to_numpy()
        )
        family_batch = frame.loc[valid, "Attack"].fillna("unknown").astype(str)
        for name in family_batch.unique():
            if name not in family_to_code:
                family_to_code[name] = len(family_to_code)
        family_codes = family_batch.map(family_to_code).to_numpy(dtype=np.int16)
        end = written + len(y_batch)
        X_store[written:end] = X_batch
        y_store[written:end] = y_batch
        family_store[written:end] = family_codes
        written = end
        print(f"aligned confirmation rows={written:,}", flush=True)
    X_store.flush(); y_store.flush(); family_store.flush()
    del X_store, y_store, family_store
    metadata = {
        "allocated_rows": total_rows, "valid_rows": written,
        "feature_order": feature_order, "family_to_code": family_to_code,
    }
    write_json(metadata_path, metadata)
    write_json(output["manifests"] / "confirmation_dataset.json", {
        "dataset": "CIC-ToN-IoT V2",
        "official_doi": "10.48610/f6884ce",
        "official_page": "https://staff.itee.uq.edu.au/marius/NIDS_datasets/",
        "download_mirror": "https://www.kaggle.com/datasets/dhoogla/cictoniot",
        "archive_sha256": ARCHIVE_SHA256,
        "raw_parquet_rows": parquet.metadata.num_rows,
        "valid_aligned_rows": int(written),
        "features_aligned": len(feature_order),
        "labels_used_for_training_or_thresholds": False,
        "note": (
            "The mirror's cleaned V2 has a different row count from the earlier "
            "official-page release description; no rows were selected by label."
        ),
    })
    return cache


def family_error_rows(seed: int, family, family_names, y, mask, prediction):
    rows = []
    for code in np.unique(family):
        selected = mask & (family == code)
        if not selected.any():
            continue
        actual = y[selected]
        predicted = prediction[selected]
        rows.append({
            "seed": seed, "attack_family": family_names[int(code)],
            "covered": int(selected.sum()),
            "false_positives": int(((actual == 0) & (predicted == 1)).sum()),
            "false_negatives": int(((actual == 1) & (predicted == 0)).sum()),
        })
    return rows


def batched_probabilities(model, X, destination: Path, batch_size: int = 100_000):
                                                                                
    probability = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(len(X),)
    )
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        probability[start:end] = model.predict_proba(X[start:end])[:, 1]
    probability.flush()
    del probability
    return np.load(destination, mmap_mode="r")


def streaming_binary_metrics(y, prediction, probability, mask=None, bins=16_384):
                                                                               

                                                                              
                                                                            
                                                   
       
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    tn = fp = fn = tp = count = 0
    squared_error = negative_log_likelihood = 0.0
    positive_hist = np.zeros(bins, dtype=np.int64)
    negative_hist = np.zeros(bins, dtype=np.int64)
    calibration_count = np.zeros(10, dtype=np.int64)
    calibration_correct = np.zeros(10, dtype=np.float64)
    calibration_confidence = np.zeros(10, dtype=np.float64)
    for start in range(0, len(y), 100_000):
        end = min(start + 100_000, len(y))
        selected = mask[start:end]
        if not selected.any():
            continue
        actual = np.asarray(y[start:end][selected], dtype=np.int8)
        predicted = np.asarray(prediction[start:end][selected], dtype=np.int8)
        score = np.clip(
            np.asarray(probability[start:end][selected], dtype=np.float64),
            1e-7, 1 - 1e-7,
        )
        count += len(actual)
        tn += int(((actual == 0) & (predicted == 0)).sum())
        fp += int(((actual == 0) & (predicted == 1)).sum())
        fn += int(((actual == 1) & (predicted == 0)).sum())
        tp += int(((actual == 1) & (predicted == 1)).sum())
        squared_error += float(np.square(actual - score).sum())
        negative_log_likelihood += float(
            -(actual * np.log(score) + (1 - actual) * np.log(1 - score)).sum()
        )
        score_bin = np.minimum((score * bins).astype(np.int64), bins - 1)
        positive_hist += np.bincount(
            score_bin[actual == 1], minlength=bins
        )
        negative_hist += np.bincount(
            score_bin[actual == 0], minlength=bins
        )
        confidence = np.where(predicted == 1, score, 1 - score)
        calibration_bin = np.minimum(
            (confidence * 10).astype(np.int64), 9
        )
        calibration_count += np.bincount(calibration_bin, minlength=10)
        calibration_correct += np.bincount(
            calibration_bin, weights=(predicted == actual), minlength=10
        )
        calibration_confidence += np.bincount(
            calibration_bin, weights=confidence, minlength=10
        )
    positives, negatives = positive_hist.sum(), negative_hist.sum()
    negative_below = np.cumsum(negative_hist) - negative_hist
    roc_auc = float(
        np.sum(positive_hist * (negative_below + 0.5 * negative_hist))
        / max(1, positives * negatives)
    )
    descending_positive = positive_hist[::-1]
    descending_negative = negative_hist[::-1]
    cumulative_positive = np.cumsum(descending_positive)
    cumulative_negative = np.cumsum(descending_negative)
    precision_curve = cumulative_positive / np.maximum(
        1, cumulative_positive + cumulative_negative
    )
    pr_auc = float(
        np.sum(precision_curve * descending_positive) / max(1, positives)
    )
    valid_calibration = calibration_count > 0
    calibration_accuracy = np.divide(
        calibration_correct, calibration_count,
        out=np.zeros(10), where=valid_calibration,
    )
    mean_confidence = np.divide(
        calibration_confidence, calibration_count,
        out=np.zeros(10), where=valid_calibration,
    )
    ece = float(np.sum(
        calibration_count * np.abs(calibration_accuracy - mean_confidence)
    ) / max(1, count))
    accuracy = (tp + tn) / max(1, count)
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * tpr / max(1e-15, precision + tpr)
    denominator = np.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "n_samples": int(count), "accuracy": float(accuracy),
        "balanced_accuracy": float((tpr + specificity) / 2),
        "precision": float(precision), "recall": float(tpr), "f1": float(f1),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "mcc": float(mcc),
        "tpr": float(tpr), "fpr": float(fpr), "ece": ece,
        "brier": squared_error / max(1, count),
        "negative_log_likelihood": negative_log_likelihood / max(1, count),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "auc_method": f"all-row fixed histogram ({bins} bins)",
    }


def _confirmation_cache():
    cache = build_aligned_cache()
    metadata = json.loads((cache / "metadata.json").read_text())
    valid_rows = int(metadata["valid_rows"])
    return cache, metadata, valid_rows


def prepare_seed(seed: int):
                                                                    
    output = paths()
    target = output["models"] / f"confirmation_branches_seed_{seed}.pkl"
    if target.exists():
        return target
    X_train, y_train = domain_balanced_sample(DOMAIN_NAMES, seed)
    models, training_seconds = {}, 0.0
    for name in ("decision_tree", "xgboost"):
        set_seed(seed)
        model = tuned_pipeline(name, seed)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds += time.perf_counter() - started
        models[name] = model
    with target.open("wb") as handle:
        pickle.dump({"models": models, "training_seconds": training_seconds}, handle)
    print(f"confirmation seed={seed} branches prepared", flush=True)
    return target


def cache_seed_domain_sample(seed: int, domain: str):
    output = paths()
    directory = output["data"] / "confirmation_training_samples" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    domain_directory = directory / domain
    domain_directory.mkdir(parents=True, exist_ok=True)
    X_target, y_target = domain_directory / "X.npy", domain_directory / "y.npy"
    legacy = directory / f"{domain}.npz"
    if not X_target.exists():
        if legacy.exists():
            with np.load(legacy) as data:
                np.save(X_target, data["X"])
                np.save(y_target, data["y"])
            return X_target
        X, y = domain_balanced_sample([domain], seed)
        np.save(X_target, X)
        np.save(y_target, y)
        print(
            f"confirmation seed={seed} domain={domain} sample cached",
            flush=True,
        )
    return X_target


def prepare_seed_transform(seed: int):
    output = paths()
    directory = output["data"] / "confirmation_training_samples" / f"seed_{seed}"
    X_target, y_target = directory / "X_transformed.npy", directory / "y.npy"
    transformer_target = output["models"] / f"confirmation_preprocessor_seed_{seed}.pkl"
    if X_target.exists() and y_target.exists() and transformer_target.exists():
        return X_target
    X_parts, y_parts = [], []
    for domain in DOMAIN_NAMES:
        cache_seed_domain_sample(seed, domain)
        domain_directory = directory / domain
        X_parts.append(np.load(domain_directory / "X.npy", mmap_mode="r"))
        y_parts.append(np.load(domain_directory / "y.npy", mmap_mode="r"))
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    order = np.random.default_rng(seed).permutation(len(y_train))
    transformer = selected_transformer(seed)
    X_transformed = transformer.fit_transform(X_train[order])
    np.save(X_target, X_transformed)
    np.save(y_target, y_train[order])
    with transformer_target.open("wb") as handle:
        pickle.dump(transformer, handle)
    print(f"confirmation seed={seed} shared transform prepared", flush=True)
    return X_target


def prepare_seed_branch(seed: int, branch: str):
    output = paths()
    model_path = output["models"] / f"confirmation_{branch}_seed_{seed}.pkl"
    timing_path = output["models"] / f"confirmation_{branch}_seed_{seed}.json"
    if model_path.exists() and timing_path.exists():
        return model_path
    directory = output["data"] / "confirmation_training_samples" / f"seed_{seed}"
    X_train = np.load(directory / "X_transformed.npy", mmap_mode="r")
    y_train = np.load(directory / "y.npy", mmap_mode="r")
    set_seed(seed)
    model, _ = model_and_space(branch, seed)
    tuning = json.loads(
        (output["results"] / f"group_tuning_{branch}.json").read_text()
    )["best_parameters"]
    model.set_params(**{
        key.removeprefix("model__"): value
        for key, value in tuning.items() if key.startswith("model__")
    })
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    with (output["models"] / f"confirmation_preprocessor_seed_{seed}.pkl").open("rb") as handle:
        transformer = pickle.load(handle)
    pipeline = Pipeline([("preprocessor", transformer), ("model", model)])
    with model_path.open("wb") as handle:
        pickle.dump(pipeline, handle)
    write_json(timing_path, {"training_seconds": training_seconds})
    print(f"confirmation seed={seed} branch={branch} prepared", flush=True)
    return model_path


def score_seed_branch(seed: int, branch: str):
                                                                               
    output = paths()
    cache, _, valid_rows = _confirmation_cache()
    X = np.load(cache / "X.npy", mmap_mode="r")[:valid_rows]
    work = output["data"] / "confirmation_probability_cache" / f"seed_{seed}"
    work.mkdir(parents=True, exist_ok=True)
    target = work / f"{branch}.npy"
    timing = work / f"{branch}_timing.json"
    progress_path = work / f"{branch}_progress.json"
    if target.exists() and timing.exists():
        return target
    single_model = output["models"] / f"confirmation_{branch}_seed_{seed}.pkl"
    if single_model.exists():
        with single_model.open("rb") as handle:
            model = pickle.load(handle)
    else:
        with (output["models"] / f"confirmation_branches_seed_{seed}.pkl").open("rb") as handle:
            model = pickle.load(handle)["models"][branch]
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        start_row = int(progress["next_row"])
        elapsed = float(progress["inference_seconds"])
        probability = np.lib.format.open_memmap(
            target, mode="r+", dtype=np.float32, shape=(len(X),)
        )
    else:
        start_row, elapsed = 0, 0.0
        probability = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.float32, shape=(len(X),)
        )
    stop_row = min(start_row + 1_500_000, len(X))
    started = time.perf_counter()
    for start in range(start_row, stop_row, 100_000):
        end = min(start + 100_000, stop_row)
        probability[start:end] = model.predict_proba(X[start:end])[:, 1]
    elapsed += time.perf_counter() - started
    probability.flush(); del probability
    if stop_row == len(X):
        write_json(timing, {"inference_seconds": elapsed})
        print(f"confirmation seed={seed} branch={branch} scored", flush=True)
    else:
        write_json(progress_path, {
            "next_row": stop_row, "inference_seconds": elapsed,
        })
        print(
            f"confirmation seed={seed} branch={branch} "
            f"scored_rows={stop_row}/{len(X)}",
            flush=True,
        )
    return target


def finalize_seed(seed: int):
                                                                                    
    output = paths()
    result_path = output["results"] / "confirmation_ten_seed_runs.csv"
    existing = (
        pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    )
    if not existing.empty and seed in existing["seed"].astype(int).unique():
        return existing
    cache, metadata, valid_rows = _confirmation_cache()
    work = output["data"] / "confirmation_probability_cache" / f"seed_{seed}"
    dt = np.load(work / "decision_tree.npy", mmap_mode="r")
    xgb = np.load(work / "xgboost.npy", mmap_mode="r")
    selective = predict_selective(dt, xgb)
    mean_path = work / "mean_probability.npy"
    mean_store = np.lib.format.open_memmap(
        mean_path, mode="w+", dtype=np.float32, shape=(valid_rows,)
    )
    for start in range(0, valid_rows, 100_000):
        end = min(start + 100_000, valid_rows)
        mean_store[start:end] = (dt[start:end] + xgb[start:end]) / 2
    mean_store.flush(); del mean_store
    mean_probability = np.load(mean_path, mmap_mode="r")

                                                                           
    y = np.load(cache / "y.npy", mmap_mode="r")[:valid_rows]
    family = np.load(cache / "family.npy", mmap_mode="r")[:valid_rows]
    family_names = {
        int(code): name for name, code in metadata["family_to_code"].items()
    }
    combined_bundle = output["models"] / f"confirmation_branches_seed_{seed}.pkl"
    if combined_bundle.exists():
        with combined_bundle.open("rb") as handle:
            bundle = pickle.load(handle)
        models = bundle["models"]
        training_seconds = bundle["training_seconds"]
    else:
        models = {}
        training_seconds = 0.0
        for branch in ("decision_tree", "xgboost"):
            with (output["models"] / f"confirmation_{branch}_seed_{seed}.pkl").open("rb") as handle:
                models[branch] = pickle.load(handle)
            training_seconds += json.loads(
                (output["models"] / f"confirmation_{branch}_seed_{seed}.json").read_text()
            )["training_seconds"]
    inference_seconds = sum(
        json.loads((work / f"{branch}_timing.json").read_text())["inference_seconds"]
        for branch in ("decision_tree", "xgboost")
    )
    rows, errors = [], []
    for variant, mask, prediction, probability in (
        ("adaptive_decision_tree", np.ones(valid_rows, dtype=bool),
         selective["dt_prediction"], dt),
        ("selective_agreement_hybrid", selective["covered"],
         selective["dt_prediction"], mean_probability),
    ):
        metrics = streaming_binary_metrics(y, prediction, probability, mask)
        rows.append({
            "seed": seed, "variant": variant, "coverage": float(mask.mean()),
            "abstained": int((~mask).sum()),
            "abstained_attacks": int(((~mask) & (y == 1)).sum()),
            "dt_threshold": selective["dt_threshold"],
            "xgb_threshold": selective["xgb_threshold"],
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds, **metrics,
        })
        errors.extend(family_error_rows(
            seed, family, family_names, y, mask, prediction
        ))
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(result_path, index=False)
    error_path = output["tables"] / "confirmation_family_errors.csv"
    old_errors = pd.read_csv(error_path) if error_path.exists() else pd.DataFrame()
    pd.concat([old_errors, pd.DataFrame(errors)], ignore_index=True).to_csv(
        error_path, index=False
    )
    if seed == 42:
        prediction_dir = output["results"] / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_dir / "confirmation_seed42.npz",
            y=y, family=family,
            family_names=np.asarray(
                [family_names[index] for index in sorted(family_names)], dtype="U32"
            ),
            dt_probability=np.asarray(dt, dtype=np.float32),
            xgb_probability=np.asarray(xgb, dtype=np.float32),
            covered=selective["covered"],
            selective_prediction=selective["selective_prediction"],
        )
        with (output["models"] / "confirmation_frozen_seed42.pkl").open("wb") as handle:
            pickle.dump(models, handle)
    print(f"confirmation seed={seed} finalized", flush=True)
    return combined


def finish_confirmation():
    output = paths()
    result = pd.read_csv(output["results"] / "confirmation_ten_seed_runs.csv")
    expected = sorted(config()["study"]["seeds"])
    completed = sorted(result["seed"].unique().astype(int).tolist())
    if completed != expected:
        raise RuntimeError(f"Expected seeds {expected}, found {completed}")
    summary = result.groupby("variant")[
        ["coverage", "accuracy", "precision", "recall", "f1", "roc_auc",
         "pr_auc", "mcc", "tpr", "fpr", "ece", "brier",
         "training_seconds", "inference_seconds"]
    ].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(
        output["tables"] / "confirmation_mean_std.csv", index=False
    )
    write_json(output["manifests"] / "confirmation_execution.json", {
        "method_frozen_before_access": True, "labels_used_for_revision": False,
        "seeds_completed": completed, "rows": int(result["n_samples"].max()),
        "status": "ONE_SHOT_CONFIRMATION_COMPLETE",
    })
    return summary


def run(force: bool = False, seeds: list[int] | None = None):
    output = paths()
    frozen = json.loads(
        (output["manifests"] / "final_method_freeze.json").read_text()
    )
    if frozen["status"] != "FINAL_METHOD_FROZEN_BEFORE_CONFIRMATION_DOWNLOAD":
        raise RuntimeError("Final method was not frozen before confirmation access")
    cache = build_aligned_cache()
    result_path = output["results"] / "confirmation_ten_seed_runs.csv"
    rows = [] if force or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    completed = {int(row["seed"]) for row in rows}
    error_path = output["tables"] / "confirmation_family_errors.csv"
    errors = (
        [] if force or not error_path.exists()
        else pd.read_csv(error_path).to_dict("records")
    )
    metadata = json.loads((cache / "metadata.json").read_text())
    valid_rows = int(metadata["valid_rows"])
    X = np.load(cache / "X.npy", mmap_mode="r")[:valid_rows]
    work_root = output["data"] / "confirmation_probability_cache"
    work_root.mkdir(parents=True, exist_ok=True)
    family_names = {
        int(code): name for name, code in metadata["family_to_code"].items()
    }
                                                                              
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
        seed_work = work_root / f"seed_{seed}"
        seed_work.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        dt = batched_probabilities(
            models["decision_tree"], X, seed_work / "decision_tree.npy"
        )
        xgb = batched_probabilities(
            models["xgboost"], X, seed_work / "xgboost.npy"
        )
        inference_seconds = time.perf_counter() - started
        selective = predict_selective(dt, xgb)

                                                                                   
        y = np.load(cache / "y.npy", mmap_mode="r")[:valid_rows]
        family = np.load(cache / "family.npy", mmap_mode="r")[:valid_rows]
        mean_probability_store = np.lib.format.open_memmap(
            seed_work / "mean_probability.npy", mode="w+",
            dtype=np.float32, shape=(len(y),),
        )
        for start in range(0, len(y), 100_000):
            end = min(start + 100_000, len(y))
            mean_probability_store[start:end] = (
                dt[start:end] + xgb[start:end]
            ) / 2
        mean_probability_store.flush()
        del mean_probability_store
        mean_probability = np.load(
            seed_work / "mean_probability.npy", mmap_mode="r"
        )
        variants = (
            ("adaptive_decision_tree", np.ones(len(y), dtype=bool),
             selective["dt_prediction"], dt),
            ("selective_agreement_hybrid", selective["covered"],
             selective["dt_prediction"], mean_probability),
        )
        for variant, mask, prediction, probability in variants:
            metrics = streaming_binary_metrics(
                y, prediction, probability, mask=mask
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
            errors.extend(family_error_rows(
                seed, family, family_names, y, mask, prediction
            ))
        if seed == 42:
            prediction_dir = output["results"] / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_dir / "confirmation_seed42.npz",
                y=y, family=family, family_names=np.asarray(
                    [family_names[index] for index in sorted(family_names)],
                    dtype="U32",
                ), dt_probability=dt.astype(np.float32),
                xgb_probability=xgb.astype(np.float32),
                covered=selective["covered"],
                selective_prediction=selective["selective_prediction"],
            )
            with (output["models"] / "confirmation_frozen_seed42.pkl").open("wb") as handle:
                pickle.dump(models, handle)
        pd.DataFrame(rows).to_csv(result_path, index=False)
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"confirmation seed={seed} complete", flush=True)

    result = pd.DataFrame(rows)
    summary = result.groupby("variant")[
        ["coverage", "accuracy", "precision", "recall", "f1", "roc_auc",
         "pr_auc", "mcc", "tpr", "fpr", "ece", "brier",
         "training_seconds", "inference_seconds"]
    ].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(
        output["tables"] / "confirmation_mean_std.csv", index=False
    )
    pd.DataFrame(errors).to_csv(error_path, index=False)
    completed_seeds = sorted(result["seed"].unique().astype(int).tolist())
    expected_seeds = sorted(config()["study"]["seeds"])
    if completed_seeds == expected_seeds:
        write_json(output["manifests"] / "confirmation_execution.json", {
            "method_frozen_before_access": True,
            "labels_used_for_revision": False,
            "seeds_completed": completed_seeds,
            "rows": int(result["n_samples"].max()),
            "status": "ONE_SHOT_CONFIRMATION_COMPLETE",
        })
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--phase",
        choices=(
            "all", "sample-domain", "prepare", "prepare-dt", "prepare-xgb",
            "transform", "score-dt", "score-xgb", "finalize", "finish",
        ),
        default="all",
    )
    parser.add_argument("--domain", choices=DOMAIN_NAMES)
    arguments = parser.parse_args()
    if arguments.phase == "all":
        print(run(seeds=arguments.seed).groupby("variant")[
            ["coverage", "mcc", "tpr", "fpr"]
        ].agg(["mean", "std"]))
    elif arguments.phase == "finish":
        print(finish_confirmation())
    else:
        if not arguments.seed or len(arguments.seed) != 1:
            parser.error("exactly one --seed is required for this phase")
        seed = arguments.seed[0]
        if arguments.phase == "prepare":
            prepare_seed(seed)
        elif arguments.phase == "sample-domain":
            if not arguments.domain:
                parser.error("--domain is required for sample-domain")
            cache_seed_domain_sample(seed, arguments.domain)
        elif arguments.phase == "prepare-dt":
            prepare_seed_branch(seed, "decision_tree")
        elif arguments.phase == "prepare-xgb":
            prepare_seed_branch(seed, "xgboost")
        elif arguments.phase == "transform":
            prepare_seed_transform(seed)
        elif arguments.phase == "score-dt":
            score_seed_branch(seed, "decision_tree")
        elif arguments.phase == "score-xgb":
            score_seed_branch(seed, "xgboost")
        else:
            finalize_seed(seed)
