from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.preprocessing import MinMaxScaler
from benchmark.common import resolve, row_hashes, write_json
from benchmark.supervised import load_partition
from benchmark.unknown_attack import build_family_dataset
from sklearn.model_selection import train_test_split
FEATURES = [
    "byte_count", "bytes_per_packet", "log_byte_count", "ack_flag_count",
    "flag_sum", "ack_ratio", "fin_ratio", "syn_ratio", "packet_rate",
    "log_packet_rate",
]
UNSW_COLUMNS = {
    "Total Length of Fwd Packet": "_byte_count",
    "Total Fwd Packet": "_packet_count",
    "Flow Packets/s": "packet_rate",
    "ACK Flag Count": "ack_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "FIN Flag Count": "fin_flag_count",
}
LABELS_UNSW = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
}
def _config() -> dict:
    with resolve("benchmark/readiness_config.yaml").open() as handle:
        return yaml.safe_load(handle)
def _dirs() -> dict[str, Path]:
    paths = {key: resolve(value) for key, value in _config()["paths"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
def align_cic_unsw() -> dict:
    paths = _dirs()
    data_path = resolve("data/raw/cic_unsw_nb15/Data.csv")
    label_path = resolve("data/raw/cic_unsw_nb15/Lable.csv")
    usecols = list(UNSW_COLUMNS)
    data = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    labels = pd.read_csv(label_path)["Label"]
    if len(data) != len(labels):
        raise ValueError("CIC-UNSW feature and label row counts differ")
    data.rename(columns=UNSW_COLUMNS, inplace=True)
    data["attack_family_id"] = labels.astype("int8")
    for column in UNSW_COLUMNS.values():
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    eps = 1e-9
    data["_byte_count"] = data["_byte_count"].clip(lower=0)
    data["_packet_count"] = data["_packet_count"].clip(lower=0)
    data["packet_rate"] = data["packet_rate"].clip(lower=0)
    for flag in ("ack_flag_count", "syn_flag_count", "fin_flag_count"):
        data[flag] = data[flag].clip(lower=0)
    data["byte_count"] = data["_byte_count"]
    data["bytes_per_packet"] = data["_byte_count"] / (data["_packet_count"] + eps)
    data["log_byte_count"] = np.log1p(data["_byte_count"])
    data["flag_sum"] = (
        data["ack_flag_count"] + data["syn_flag_count"] + data["fin_flag_count"]
    )
    data["ack_ratio"] = data["ack_flag_count"] / (data["_packet_count"] + eps)
    data["fin_ratio"] = data["fin_flag_count"] / (data["_packet_count"] + eps)
    data["syn_ratio"] = data["syn_flag_count"] / (data["_packet_count"] + eps)
    data["log_packet_rate"] = np.log1p(data["packet_rate"])
    data["binary_label"] = (data["attack_family_id"] != 0).astype("int8")
    before = len(data)
    family_before = data["attack_family_id"].value_counts().sort_index()
    data.drop_duplicates(
        subset=FEATURES + ["binary_label"], inplace=True, ignore_index=True
    )
    family_after = data["attack_family_id"].value_counts().sort_index()
    X = data[FEATURES].to_numpy(dtype=np.float32)
    np.savez_compressed(
        paths["data"] / "cic_unsw_nb15_external.npz",
        X=X,
        y=data["binary_label"].to_numpy(dtype=np.int8),
        family=data["attack_family_id"].to_numpy(dtype=np.int8),
        features=np.asarray(FEATURES),
        row_hash=row_hashes(X),
    )
    statistics = pd.DataFrame(
        [
            {
                "family_id": family_id,
                "family": LABELS_UNSW[family_id],
                "rows_before_dedup": int(family_before.get(family_id, 0)),
                "rows_after_dedup": int(family_after.get(family_id, 0)),
                "duplicate_fraction": float(
                    1 - family_after.get(family_id, 0) / family_before.get(family_id, 1)
                ),
            }
            for family_id in LABELS_UNSW
        ]
    )
    statistics.to_csv(paths["tables"] / "cic_unsw_dataset_statistics.csv", index=False)
    manifest = {
        "official_provenance": "https://www.unb.ca/cic/datasets/cic-unsw-nb15.html",
        "mirror": "https://huggingface.co/datasets/bencorn/CIC-UNSW-NB15",
        "data_sha256": _sha256(data_path),
        "label_sha256": _sha256(label_path),
        "rows_before_dedup": before,
        "rows_after_dedup": len(data),
        "features": FEATURES,
        "binary_mapping": "0=Benign; 1-9=Attack",
    }
    write_json(paths["manifests"] / "cic_unsw_manifest.json", manifest)
    return manifest
def cse_duplicate_audit() -> pd.DataFrame:
    paths = _dirs()
    source = pd.read_csv(
        resolve("data/processed/cic_ids2018/cic2018_aligned_features.csv"),
        usecols=FEATURES + ["label"],
        low_memory=False,
    )
    source.replace([np.inf, -np.inf], np.nan, inplace=True)
    source.dropna(inplace=True)
    source["binary_label"] = (source["label"] != 0).astype("int8")
    before = source.groupby("label").size()
    unique = source.drop_duplicates(subset=FEATURES + ["binary_label"])
    after = unique.groupby("label").size()
    names = {0: "Benign", 1: "FTP-BruteForce", 2: "SSH-Bruteforce"}
    table = pd.DataFrame(
        [
            {
                "label": int(label),
                "family": names.get(int(label), str(label)),
                "rows_before": int(before.get(label, 0)),
                "rows_after": int(after.get(label, 0)),
                "duplicates_removed": int(before.get(label, 0) - after.get(label, 0)),
                "duplicate_fraction": float(1 - after.get(label, 0) / before.get(label, 1)),
            }
            for label in sorted(before.index)
        ]
    )
    table.to_csv(paths["tables"] / "cse_cic_ids2018_duplicate_audit.csv", index=False)
    return table
def scaling_audit() -> pd.DataFrame:
    paths = _dirs()
    X_train, _, features = load_partition("cicids2017_train")
    datasets = {
        "cicids2017_validation": load_partition("cicids2017_validation")[0],
        "cicids2017_internal": load_partition("cicids2017_test")[0],
        "cic_ids2018_external": load_partition("cic_ids2018_external")[0],
    }
    unsw = np.load(paths["data"] / "cic_unsw_nb15_external.npz")
    datasets["cic_unsw_nb15_external"] = unsw["X"]
    scaler = MinMaxScaler().fit(X_train)
    rows = []
    for dataset, X in datasets.items():
        transformed = scaler.transform(X)
        for index, feature in enumerate(features):
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "train_min": float(X_train[:, index].min()),
                    "train_max": float(X_train[:, index].max()),
                    "dataset_min": float(X[:, index].min()),
                    "dataset_max": float(X[:, index].max()),
                    "outside_training_range": float(
                        (
                            (X[:, index] < X_train[:, index].min())
                            | (X[:, index] > X_train[:, index].max())
                        ).mean()
                    ),
                    "scaled_below_zero": float((transformed[:, index] < 0).mean()),
                    "scaled_above_one": float((transformed[:, index] > 1).mean()),
                    "scaled_p01": float(np.quantile(transformed[:, index], 0.01)),
                    "scaled_p50": float(np.quantile(transformed[:, index], 0.50)),
                    "scaled_p99": float(np.quantile(transformed[:, index], 0.99)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(paths["tables"] / "feature_scaling_audit.csv", index=False)
    return table
def score_diagnostics() -> pd.DataFrame:
    paths = _dirs()
    frames = []
    for dataset in ("cicids2017_internal", "cic_ids2018_external"):
        supervised = np.load(
            resolve(
                f"archive/benchmark_v1/results/predictions/random_forest_{dataset}_seed_42.npz"
            )
        )
        hybrid = np.load(
            resolve(
                f"archive/benchmark_v1/results/hybrid_predictions/adaptive_hybrid_{dataset}_seed_42.npz"
            )
        )
        for label in (0, 1):
            mask = supervised["y_true"] == label
            for score_name, values in (
                ("rf_attack_probability", supervised["probability"]),
                ("anomaly_probability", hybrid["anomaly_probability"]),
                ("reconstruction_error", hybrid["reconstruction_error"]),
                ("hybrid_score", hybrid["score"]),
            ):
                subset = values[mask]
                frames.append(
                    {
                        "dataset": dataset,
                        "class": "attack" if label else "benign",
                        "score": score_name,
                        "n": len(subset),
                        "mean": float(subset.mean()),
                        "std": float(subset.std()),
                        "p01": float(np.quantile(subset, 0.01)),
                        "p25": float(np.quantile(subset, 0.25)),
                        "p50": float(np.quantile(subset, 0.50)),
                        "p75": float(np.quantile(subset, 0.75)),
                        "p99": float(np.quantile(subset, 0.99)),
                    }
                )
    table = pd.DataFrame(frames)
    table.to_csv(paths["tables"] / "score_distribution_summary.csv", index=False)
    return table
def semantic_alignment_table() -> pd.DataFrame:
    paths = _dirs()
    rows = [
        ("byte_count", "Total Length of Fwd Packets", "TotLen Fwd Pkts",
         "Total Length of Fwd Packet", "bytes", "forward", "exact"),
        ("bytes_per_packet", "derived", "derived", "derived",
         "bytes/packet", "forward", "exact formula"),
        ("log_byte_count", "log1p derived", "log1p derived", "log1p derived",
         "log(bytes)", "forward", "exact formula"),
        ("ack_flag_count", "ACK Flag Count", "ACK Flag Cnt", "ACK Flag Count",
         "count", "bidirectional flow", "exact name/semantics"),
        ("flag_sum", "derived", "derived", "derived", "count",
         "bidirectional flow", "exact formula"),
        ("ack_ratio", "derived", "derived", "derived", "ratio",
         "ACK flow count / forward packets", "derived; directional caveat"),
        ("fin_ratio", "derived", "derived", "derived", "ratio",
         "FIN flow count / forward packets", "derived; directional caveat"),
        ("syn_ratio", "derived", "derived", "derived", "ratio",
         "SYN flow count / forward packets", "derived; directional caveat"),
        ("packet_rate", "Flow Packets/s", "Flow Pkts/s", "Flow Packets/s",
         "packets/second", "bidirectional flow", "exact"),
        ("log_packet_rate", "log1p derived", "log1p derived", "log1p derived",
         "log(packets/second)", "bidirectional flow", "exact formula"),
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "canonical_feature", "CICIDS2017", "CSE_CIC_IDS2018",
            "CIC_UNSW_NB15", "unit", "directionality", "alignment_status",
        ],
    )
    table.to_csv(paths["tables"] / "semantic_feature_alignment.csv", index=False)
    return table
def build_full_cicids2017_protocol() -> dict:
    paths = _dirs()
    family_path = build_family_dataset(
        max_normal=None,
        max_per_attack=None,
        seed=42,
        output_path="archive/readiness_v2/data/cicids2017_all_families_full.npz",
    )
    data = np.load(family_path, allow_pickle=False)
    X = data["X"]
    family = data["family"].astype(str)
    y = (np.char.upper(family) != "BENIGN").astype(np.int8)
    all_hashes = row_hashes(X)
    duplicate_mask = pd.Series(all_hashes).duplicated(keep=False).to_numpy()
    contradictory_rows_removed = int(duplicate_mask.sum())
    X = X[~duplicate_mask]
    family = family[~duplicate_mask]
    y = y[~duplicate_mask]
    indices = np.arange(len(y))
    train_val, test = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=family
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=42,
        stratify=family[train_val],
    )
    partitions = {"train": train, "validation": validation, "test": test}
    hashes = {}
    for name, selected in partitions.items():
        selected_X = X[selected]
        selected_hash = row_hashes(selected_X)
        hashes[name] = set(selected_hash)
        np.savez_compressed(
            paths["data"] / f"cicids2017_multifamily_{name}.npz",
            X=selected_X.astype(np.float32),
            y=y[selected],
            family=family[selected],
            features=np.asarray(FEATURES),
            row_hash=selected_hash,
        )
    overlap = {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Multi-family split overlap: {overlap}")
    rows = []
    for name, selected in partitions.items():
        counts = pd.Series(family[selected]).value_counts()
        for attack_family, count in counts.items():
            rows.append(
                {
                    "partition": name,
                    "family": attack_family,
                    "binary_label": int(attack_family.upper() != "BENIGN"),
                    "rows": int(count),
                }
            )
    pd.DataFrame(rows).to_csv(
        paths["tables"] / "cicids2017_multifamily_statistics.csv", index=False
    )
    manifest = {
        "source": "all eight supplied CICIDS2017 daily CSV files",
        "real_data_only": True,
        "rows": len(y),
        "partitions": {name: len(selected) for name, selected in partitions.items()},
        "feature_overlap": overlap,
        "families": sorted(np.unique(family).tolist()),
        "contradictory_or_cross_family_duplicate_rows_removed": contradictory_rows_removed,
    }
    write_json(paths["manifests"] / "cicids2017_multifamily_manifest.json", manifest)
    return manifest
def run() -> None:
    align_cic_unsw()
    cse_duplicate_audit()
    scaling_audit()
    score_diagnostics()
    semantic_alignment_table()
    build_full_cicids2017_protocol()
if __name__ == "__main__":
    run()
