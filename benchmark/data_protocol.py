from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from benchmark.common import (
    ensure_output_dirs,
    environment_manifest,
    load_config,
    resolve,
    row_hashes,
    sha256_file,
    write_json,
)
def _load_aligned(path: Path, features: list[str], label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Aligned dataset not found: {path}")
    df = pd.read_csv(path, usecols=features + [label_col], low_memory=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + [label_col], inplace=True)
    df[label_col] = (pd.to_numeric(df[label_col], errors="raise") != 0).astype("int8")
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df.dropna(inplace=True)
    return df.reset_index(drop=True)
def _balanced_cap(
    df: pd.DataFrame, label_col: str, max_rows: int | None, seed: int
) -> pd.DataFrame:
    if max_rows is None:
        return df
    groups = []
    for _, group in df.groupby(label_col, sort=True):
        groups.append(group.sample(n=min(len(group), max_rows), random_state=seed))
    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)
def _save_partition(path: Path, frame: pd.DataFrame, features: list[str], label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=frame[features].to_numpy(dtype=np.float32),
        y=frame[label].to_numpy(dtype=np.int8),
        row_hash=row_hashes(frame[features].to_numpy()),
        features=np.asarray(features),
    )
def _class_counts(frame: pd.DataFrame, label: str) -> dict[str, int]:
    return {
        str(int(key)): int(value)
        for key, value in frame[label].value_counts().sort_index().items()
    }
def build(config_path: str | None = None) -> dict:
    config = load_config(config_path or resolve("benchmark/config.yaml"))
    ensure_output_dirs(config)
    data_cfg = config["data"]
    features = list(config["features"]["candidates"])
    label = data_cfg["label_column"]
    seed = int(data_cfg["random_seed"])
    output_dir = resolve(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve(data_cfg["train_domain"])
    source = _load_aligned(source_path, features, label)
    source_rows_before_dedup = len(source)
    if data_cfg.get("remove_duplicates", True):
        source = source.drop_duplicates(subset=features + [label]).reset_index(drop=True)
    contradictory = int(
        source.groupby(features, dropna=False)[label].nunique().gt(1).sum()
    )
    if contradictory:
        conflicting_hashes = source.groupby(features)[label].transform("nunique").gt(1)
        source = source.loc[~conflicting_hashes].reset_index(drop=True)
    source = _balanced_cap(
        source, label, data_cfg.get("max_rows_per_class"), seed
    )
    train_val, test = train_test_split(
        source,
        test_size=float(data_cfg["test_size"]),
        stratify=source[label],
        random_state=seed,
    )
    relative_val = float(data_cfg["validation_size"]) / (
        1.0 - float(data_cfg["test_size"])
    )
    train, validation = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label],
        random_state=seed,
    )
    partitions = {"train": train, "validation": validation, "test": test}
    hash_sets = {
        name: set(row_hashes(frame[features].to_numpy()))
        for name, frame in partitions.items()
    }
    overlap = {
        "train_validation": len(hash_sets["train"] & hash_sets["validation"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "validation_test": len(hash_sets["validation"] & hash_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Partition overlap detected: {overlap}")
    for name, frame in partitions.items():
        _save_partition(output_dir / f"cicids2017_{name}.npz", frame, features, label)
    external_manifest = {}
    train_feature_hashes = set(row_hashes(source[features].to_numpy()))
    for domain, configured_path in data_cfg["external_domains"].items():
        path = resolve(configured_path)
        external = _load_aligned(path, features, label)
        rows_before = len(external)
        if data_cfg.get("remove_duplicates", True):
            external = external.drop_duplicates(subset=features + [label]).reset_index(
                drop=True
            )
        overlap_with_source = len(
            train_feature_hashes & set(row_hashes(external[features].to_numpy()))
        )
        _save_partition(output_dir / f"{domain}_external.npz", external, features, label)
        external_manifest[domain] = {
            "source": str(path.relative_to(resolve("."))),
            "source_sha256": sha256_file(path),
            "rows_before_deduplication": rows_before,
            "rows_after_deduplication": len(external),
            "class_counts": _class_counts(external, label),
            "feature_overlap_with_training_domain": overlap_with_source,
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "CICIDS2017 train/validation/internal-test; external test only",
        "source": str(source_path.relative_to(resolve("."))),
        "source_sha256": sha256_file(source_path),
        "features": features,
        "label_mapping": {"normal": 0, "any_attack": 1},
        "random_seed": seed,
        "rows_before_deduplication": source_rows_before_dedup,
        "rows_after_cleaning": len(source),
        "contradictory_duplicate_feature_vectors_removed": contradictory,
        "partitions": {
            name: {
                "rows": len(frame),
                "class_counts": _class_counts(frame, label),
                "artifact": str((output_dir / f"cicids2017_{name}.npz").relative_to(resolve("."))),
            }
            for name, frame in partitions.items()
        },
        "partition_feature_overlap": overlap,
        "external_domains": external_manifest,
    }
    write_json(resolve("archive/benchmark_v1/manifests/data_manifest.json"), manifest)
    write_json(
        resolve("archive/benchmark_v1/manifests/environment.json"), environment_manifest()
    )
    pd.DataFrame(
        [
            {
                "dataset": "CICIDS2017",
                "partition": name,
                "rows": len(frame),
                "normal": _class_counts(frame, label).get("0", 0),
                "attack": _class_counts(frame, label).get("1", 0),
            }
            for name, frame in partitions.items()
        ]
        + [
            {
                "dataset": domain,
                "partition": "external_test",
                "rows": details["rows_after_deduplication"],
                "normal": details["class_counts"].get("0", 0),
                "attack": details["class_counts"].get("1", 0),
            }
            for domain, details in external_manifest.items()
        ]
    ).to_csv(resolve("archive/benchmark_v1/tables/dataset_statistics.csv"), index=False)
    return manifest
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/config.yaml")
    args = parser.parse_args()
    manifest = build(args.config)
    print("Phase 1 data protocol complete.")
    print(manifest["partitions"])
if __name__ == "__main__":
    main()
