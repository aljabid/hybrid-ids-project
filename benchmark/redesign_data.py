                                                                              

from __future__ import annotations

import hashlib
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmark.common import resolve, row_hashes, write_json

CANONICAL = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Max": "fwd_packet_length_max",
    "Fwd Packet Length Min": "fwd_packet_length_min",
    "Fwd Packet Length Mean": "fwd_packet_length_mean",
    "Fwd Packet Length Std": "fwd_packet_length_std",
    "Bwd Packet Length Max": "bwd_packet_length_max",
    "Bwd Packet Length Min": "bwd_packet_length_min",
    "Bwd Packet Length Mean": "bwd_packet_length_mean",
    "Bwd Packet Length Std": "bwd_packet_length_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_total",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_total",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_length",
    "Bwd Header Length": "bwd_header_length",
    "Fwd Packets/s": "fwd_packets_per_s",
    "Bwd Packets/s": "bwd_packets_per_s",
    "Min Packet Length": "packet_length_min",
    "Max Packet Length": "packet_length_max",
    "Packet Length Mean": "packet_length_mean",
    "Packet Length Std": "packet_length_std",
    "Packet Length Variance": "packet_length_variance",
    "FIN Flag Count": "fin_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "URG Flag Count": "urg_flag_count",
    "CWE Flag Count": "cwr_flag_count",
    "ECE Flag Count": "ece_flag_count",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "average_packet_size",
    "Avg Fwd Segment Size": "avg_fwd_segment_size",
    "Avg Bwd Segment Size": "avg_bwd_segment_size",
    "Subflow Fwd Packets": "subflow_fwd_packets",
    "Subflow Fwd Bytes": "subflow_fwd_bytes",
    "Subflow Bwd Packets": "subflow_bwd_packets",
    "Subflow Bwd Bytes": "subflow_bwd_bytes",
    "act_data_pkt_fwd": "fwd_active_data_packets",
    "min_seg_size_forward": "fwd_segment_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}

ALIASES_2018 = {
    "Tot Fwd Pkts": "Total Fwd Packets", "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max", "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean", "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max", "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean", "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s", "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total", "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length", "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s", "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length", "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean", "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance", "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count", "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count", "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count", "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size", "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size", "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes", "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes", "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

ALIASES_UNSW = {
    "Total Fwd Packet": "Total Fwd Packets", "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length", "Packet Length Max": "Max Packet Length",
    "Packet Length Mean": "Packet Length Mean", "Packet Length Std": "Packet Length Std",
    "Packet Length Variance": "Packet Length Variance", "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size", "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Act Data Pkts": "act_data_pkt_fwd", "Fwd Seg Size Min": "min_seg_size_forward",
}


def config():
    with resolve("benchmark/redesign_config.yaml").open() as handle:
        return yaml.safe_load(handle)


def dirs():
    result = {key: resolve(value) for key, value in config()["paths"].items()}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def _clean_chunk(chunk: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    chunk.columns = chunk.columns.str.strip()
    chunk.rename(columns=aliases, inplace=True)
    use = [column for column in CANONICAL if column in chunk]
    if set(use) != set(CANONICAL):
        missing = sorted(set(CANONICAL) - set(use))
        raise ValueError(f"Missing semantic features: {missing}")
    values = chunk[use].rename(columns=CANONICAL)
    for column in values:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values.replace([np.inf, -np.inf], np.nan, inplace=True)
    return values


def build_cicids2017(force: bool = False) -> dict:
    output = dirs()
    manifest_path = output["manifests"] / "grouped_data_manifest.json"
    if manifest_path.exists() and not force:
        import json
        return json.loads(manifest_path.read_text())
    temporary = output["data"] / "_group_cache"
    temporary.mkdir(parents=True, exist_ok=True)
    features = list(CANONICAL.values())
    group_records = []
    all_families: set[str] = set()
    raw_clean = 0
    for source in sorted(resolve("data/raw/cicids2017").glob("*.csv")):
        cache_path = temporary / f"{source.name}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            hashes = cached["hash"]
            family_group = cached["family"].astype(str)
            raw_clean += len(hashes)
            all_families.update(family_group.tolist())
            group_records.append((source.name, cache_path, hashes))
            print(f"reused {source.name}: {len(hashes):,}", flush=True)
            del cached, family_group
            gc.collect()
            continue
        arrays, labels = [], []
        for chunk in pd.read_csv(source, chunksize=25000, encoding="latin-1", low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            clean = _clean_chunk(chunk, {})
            valid = ~clean.isna().any(axis=1)
            values = clean.loc[valid].clip(lower=0).to_numpy(dtype=np.float32)
            family_values = (
                chunk.loc[valid, "Label"].astype(str).str.strip().to_numpy(dtype=str)
            )
            arrays.append(values)
            labels.append(family_values)
            raw_clean += len(values)
            del chunk, clean
            gc.collect()
        X_group = np.concatenate(arrays)
        family_group = np.concatenate(labels)
        hashes = row_hashes(X_group)
                                                         
        key = pd.DataFrame({"hash": hashes, "family": family_group})
        keep_local = ~key.duplicated(subset=["hash", "family"], keep="first").to_numpy()
        X_group, family_group, hashes = (
            X_group[keep_local], family_group[keep_local], hashes[keep_local]
        )
        all_families.update(family_group.tolist())
        np.savez(cache_path, X=X_group, family=family_group, hash=hashes)
        group_records.append((source.name, cache_path, hashes))
        print(f"cached {source.name}: {len(X_group):,}", flush=True)
        del X_group, family_group, arrays, labels
        gc.collect()
    family_names = sorted(all_families)
    family_map = {name: index for index, name in enumerate(family_names)}
                                                                           
                                                                               
    all_hashes = np.concatenate([record[2] for record in group_records])
    unique_hash, hash_counts = np.unique(all_hashes, return_counts=True)
    cross_group_or_conflicting = unique_hash[hash_counts > 1]
    del all_hashes, unique_hash, hash_counts
    partitions = {
        "train": config()["data"]["train_groups"],
        "validation": config()["data"]["validation_groups"],
        "internal_test": config()["data"]["internal_test_groups"],
    }
    stats = []
    hashes_by_partition = {}
    final_family_counts = {name: 0 for name in family_names}
    for partition, groups in partitions.items():
        X_parts, family_parts, group_parts = [], [], []
        for group_name, cache_path, _ in group_records:
            if group_name not in groups:
                continue
            cached = np.load(cache_path)
            keep = ~np.isin(cached["hash"], cross_group_or_conflicting)
            selected_family = cached["family"][keep].astype(str)
            X_parts.append(cached["X"][keep])
            family_parts.append(
                np.asarray([family_map[value] for value in selected_family], dtype=np.int16)
            )
            group_parts.append(np.full(int(keep.sum()), group_name))
            for name, count in zip(*np.unique(selected_family, return_counts=True)):
                final_family_counts[name] += int(count)
        subset_X = np.concatenate(X_parts)
        subset_family = np.concatenate(family_parts)
        subset_group = np.concatenate(group_parts)
        subset_y = np.asarray(
            [family_names[index].upper() != "BENIGN" for index in subset_family],
            dtype=np.int8,
        )
        subset_path = output["data"] / f"cicids2017_expanded_{partition}.npz"
        np.savez_compressed(
            subset_path, X=subset_X, y=subset_y, family=subset_family,
            group=subset_group, features=np.asarray(features),
        )
        hashes_by_partition[partition] = set(row_hashes(subset_X).tolist())
        stats.append({
            "partition": partition, "rows": len(subset_y),
            "benign": int((subset_y == 0).sum()),
            "attack": int((subset_y == 1).sum()),
            "families": int(np.unique(subset_family[subset_y == 1]).size),
            "groups": "|".join(groups),
        })
        print(f"wrote {partition}: {len(subset_y):,}", flush=True)
        del subset_X, subset_family, subset_group, subset_y, X_parts, family_parts, group_parts
    overlaps = {
        "train_validation": len(hashes_by_partition["train"] & hashes_by_partition["validation"]),
        "train_test": len(hashes_by_partition["train"] & hashes_by_partition["internal_test"]),
        "validation_test": len(hashes_by_partition["validation"] & hashes_by_partition["internal_test"]),
    }
    pd.DataFrame(stats).to_csv(output["tables"] / "grouped_dataset_statistics.csv", index=False)
    pd.DataFrame([
        {"family": name, "family_id": family_map[name], "rows": final_family_counts[name]}
        for name in family_names
    ]).to_csv(output["tables"] / "family_mapping.csv", index=False)
    schema = pd.DataFrame([
        {
            "canonical_feature": target, "CICIDS2017": source,
            "CSE_CIC_IDS2018": next((alias for alias, canonical in ALIASES_2018.items()
                                     if canonical == source), source),
            "CIC_UNSW_NB15": next((alias for alias, canonical in ALIASES_UNSW.items()
                                   if canonical == source), source),
            "unit": "count/rate/bytes/time as emitted by CICFlowMeter",
            "compatibility": "mapped; empirical range audit required",
        }
        for source, target in CANONICAL.items()
    ])
    schema.to_csv(output["tables"] / "semantic_feature_schema.csv", index=False)
    manifest = {
        "rows_raw_clean": raw_clean,
        "rows_final": int(sum(final_family_counts.values())),
        "cross_group_or_conflicting_feature_hashes_removed": int(
            len(cross_group_or_conflicting)
        ),
        "features": features, "feature_count": len(features),
        "family_map": family_map, "partition_hash_overlap": overlaps,
        "real_data_only": True,
    }
    write_json(manifest_path, manifest)
    return manifest


def _save_external(
    X: np.ndarray, family_text: np.ndarray, dataset: str, source: str
) -> dict:
    output = dirs()
    features = list(CANONICAL.values())
    family_text = np.asarray(family_text, dtype=str)
    benign = np.char.upper(family_text) == "BENIGN"
    y = (~benign).astype(np.int8)
    hashes = row_hashes(X)
    keys = pd.DataFrame({"hash": hashes, "family": family_text})
    keep = ~keys.duplicated(subset=["hash", "family"], keep="first").to_numpy()
    X, y, family_text, hashes = X[keep], y[keep], family_text[keep], hashes[keep]
    names = sorted(np.unique(family_text).tolist())
    mapping = {name: index for index, name in enumerate(names)}
    family = np.asarray([mapping[name] for name in family_text], dtype=np.int16)
    np.savez_compressed(
        output["data"] / f"{dataset}_expanded_diagnostic.npz",
        X=X, y=y, family=family, features=np.asarray(features),
    )
    manifest = {
        "dataset": dataset, "source": source, "rows": len(y),
        "benign": int((y == 0).sum()), "attack": int((y == 1).sum()),
        "family_map": mapping, "feature_count": len(features),
        "role": "previously observed development diagnostic; not final confirmation",
    }
    write_json(output["manifests"] / f"{dataset}_diagnostic_manifest.json", manifest)
    return manifest


def build_existing_diagnostics(force: bool = False) -> list[dict]:
    output = dirs()
    manifests = []
    cse_target = output["data"] / "cse_cic_ids2018_expanded_diagnostic.npz"
    if force or not cse_target.exists():
        arrays, labels = [], []
        source = resolve("data/raw/cic_ids2018/raw_subset_portscan.csv")
        for chunk in pd.read_csv(source, chunksize=25000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            clean = _clean_chunk(chunk, ALIASES_2018)
            valid = ~clean.isna().any(axis=1)
            arrays.append(clean.loc[valid].clip(lower=0).to_numpy(np.float32))
            labels.append(chunk.loc[valid, "Label"].astype(str).str.strip().to_numpy(str))
        manifests.append(
            _save_external(
                np.concatenate(arrays), np.concatenate(labels),
                "cse_cic_ids2018", str(source),
            )
        )
    else:
        import json
        manifests.append(json.loads(
            (output["manifests"] / "cse_cic_ids2018_diagnostic_manifest.json").read_text()
        ))
    unsw_target = output["data"] / "cic_unsw_nb15_expanded_diagnostic.npz"
    if force or not unsw_target.exists():
        arrays, valid_masks = [], []
        source = resolve("data/raw/cic_unsw_nb15/Data.csv")
        for chunk in pd.read_csv(source, chunksize=25000, low_memory=False):
            clean = _clean_chunk(chunk, ALIASES_UNSW)
            valid = ~clean.isna().any(axis=1)
            arrays.append(clean.loc[valid].clip(lower=0).to_numpy(np.float32))
            valid_masks.append(valid.to_numpy())
        X = np.concatenate(arrays)
        label_ids = pd.read_csv(
            resolve("data/raw/cic_unsw_nb15/Lable.csv")
        )["Label"].to_numpy()
        label_ids = label_ids[np.concatenate(valid_masks)]
        label_names = np.asarray(
            [
                {
                    0: "BENIGN", 1: "Analysis", 2: "Backdoor", 3: "DoS",
                    4: "Exploits", 5: "Fuzzers", 6: "Generic",
                    7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
                }[int(value)]
                for value in label_ids
            ],
            dtype=str,
        )
        manifests.append(
            _save_external(X, label_names, "cic_unsw_nb15", str(source))
        )
    else:
        import json
        manifests.append(json.loads(
            (output["manifests"] / "cic_unsw_nb15_diagnostic_manifest.json").read_text()
        ))
    return manifests


if __name__ == "__main__":
    print(build_cicids2017())
    print(build_existing_diagnostics())
