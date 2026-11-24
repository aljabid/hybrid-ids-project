from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from benchmark.common import load_config, resolve, set_seed
from benchmark.hybrid_experiments import (
    anomaly_probability,
    fixed_router,
    reconstruction_errors,
    train_autoencoder,
)
from benchmark.metrics import binary_metrics
from benchmark.model_registry import model_and_space
RAW_MAP = {
    "Total Length of Fwd Packets": "_byte_count",
    "Total Fwd Packets": "_packet_count",
    "Flow Packets/s": "packet_rate",
    "ACK Flag Count": "ack_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "FIN Flag Count": "fin_flag_count",
}
FEATURES = [
    "byte_count", "bytes_per_packet", "log_byte_count", "ack_flag_count",
    "flag_sum", "ack_ratio", "fin_ratio", "syn_ratio", "packet_rate",
    "log_packet_rate",
]
def build_family_dataset(
    max_normal: int | None = 60000,
    max_per_attack: int | None = 20000,
    seed: int = 42,
    output_path: str = "archive/benchmark_v1/data/cicids2017_attack_families.npz",
) -> Path:
    output = resolve(output_path)
    if output.exists():
        return output
    frames = []
    columns = list(RAW_MAP) + ["Label"]
    for path in sorted(resolve("data/raw/cicids2017").glob("*.csv")):
        try:
            chunks = pd.read_csv(
                path,
                usecols=lambda column: column.strip() in columns,
                chunksize=100000,
                encoding="utf-8",
                low_memory=False,
            )
        except UnicodeDecodeError:
            chunks = pd.read_csv(
                path,
                usecols=lambda column: column.strip() in columns,
                chunksize=100000,
                encoding="latin-1",
                low_memory=False,
            )
        for chunk in chunks:
            chunk.columns = chunk.columns.str.strip()
            chunk.rename(columns=RAW_MAP, inplace=True)
            required = list(RAW_MAP.values()) + ["Label"]
            if any(column not in chunk for column in required):
                continue
            for column in RAW_MAP.values():
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(subset=required, inplace=True)
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            eps = 1e-9
            chunk["byte_count"] = chunk["_byte_count"].clip(lower=0)
            packet_count = chunk["_packet_count"].clip(lower=0)
            chunk["bytes_per_packet"] = chunk["byte_count"] / (packet_count + eps)
            chunk["log_byte_count"] = np.log1p(chunk["byte_count"])
            chunk["flag_sum"] = (
                chunk["ack_flag_count"] + chunk["syn_flag_count"] + chunk["fin_flag_count"]
            )
            chunk["ack_ratio"] = chunk["ack_flag_count"] / (packet_count + eps)
            chunk["fin_ratio"] = chunk["fin_flag_count"] / (packet_count + eps)
            chunk["syn_ratio"] = chunk["syn_flag_count"] / (packet_count + eps)
            chunk["packet_rate"] = chunk["packet_rate"].clip(lower=0)
            chunk["log_packet_rate"] = np.log1p(chunk["packet_rate"])
            frames.append(chunk[FEATURES + ["Label"]])
    data = pd.concat(frames, ignore_index=True)
    data.drop_duplicates(subset=FEATURES + ["Label"], inplace=True)
    rng = np.random.default_rng(seed)
    selected = []
    for label, group in data.groupby("Label"):
        cap = max_normal if label.upper() == "BENIGN" else max_per_attack
        selected.append(
            group
            if cap is None
            else group.sample(n=min(len(group), cap), random_state=seed)
        )
    data = pd.concat(selected).sample(frac=1, random_state=seed).reset_index(drop=True)
    np.savez_compressed(
        output,
        X=data[FEATURES].to_numpy(dtype=np.float32),
        family=data["Label"].to_numpy(dtype=str),
        features=np.asarray(FEATURES),
    )
    pd.DataFrame(
        {"attack_family": data["Label"].value_counts().index, "rows": data["Label"].value_counts().values}
    ).to_csv(resolve("archive/benchmark_v1/tables/attack_family_statistics.csv"), index=False)
    return output
def run(config_path: str = "benchmark/config.yaml") -> pd.DataFrame:
    config = load_config(resolve(config_path))
    data = np.load(build_family_dataset(), allow_pickle=False)
    X, family = data["X"], data["family"].astype(str)
    normal_mask = np.char.upper(family) == "BENIGN"
    counts = pd.Series(family[~normal_mask]).value_counts()
    suitable = counts[counts >= 500].head(5).index.tolist()
    tuning = [
        json.loads(path.read_text())
        for path in resolve("archive/benchmark_v1/results").glob("tuning_*.json")
    ]
    if not tuning:
        raise FileNotFoundError("Tuning results are required.")
    best = max(tuning, key=lambda row: row["best_cv_mcc"])
    model_name = best["model"]
    checkpoint = resolve("archive/benchmark_v1/results/unknown_attack_seed_results.csv")
    rows = []
    completed_seeds: set[int] = set()
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        rows.extend(existing.to_dict("records"))
        completed_seeds = set(existing["seed"].astype(int).unique())
    for seed in config["experiments"]["seeds"]:
        seed = int(seed)
        if seed in completed_seeds:
            continue
        seed_rows = []
        set_seed(seed)
        normal_indices = np.flatnonzero(normal_mask)
        normal_train, normal_test = train_test_split(
            normal_indices, test_size=0.25, random_state=seed
        )
        rng = np.random.default_rng(seed)
        normal_fit = (
            rng.choice(normal_train, 30000, replace=False)
            if len(normal_train) > 30000
            else normal_train
        )
        scaler = MinMaxScaler().fit(X[normal_fit])
        ae_indices = (
            rng.choice(normal_fit, 10000, replace=False)
            if len(normal_fit) > 10000
            else normal_fit
        )
        autoencoder, _, _ = train_autoencoder(
            scaler.transform(X[ae_indices]), seed, epochs=12
        )
        reference_indices = ae_indices[: max(1000, len(ae_indices) // 5)]
        reference = reconstruction_errors(
            autoencoder, scaler.transform(X[reference_indices])
        )
        for held_out in suitable:
            held_mask = family == held_out
            known_attack_indices = np.flatnonzero(~normal_mask & ~held_mask)
            if not len(known_attack_indices):
                continue
            if len(known_attack_indices) > 30000:
                known_attack_indices = rng.choice(
                    known_attack_indices, 30000, replace=False
                )
            train_indices = np.concatenate([normal_fit, known_attack_indices])
            y_train = (~normal_mask[train_indices]).astype(np.int8)
            positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            estimator, _ = model_and_space(model_name, seed, positive_weight)
            parameters = {
                key.replace("model__", ""): value
                for key, value in best["best_parameters"].items()
                if not key.startswith("selector__") and not key.startswith("scaler__")
            }
            estimator.set_params(**parameters)
            if model_name == "random_forest":
                estimator.set_params(n_estimators=50)
            estimator.fit(scaler.transform(X[train_indices]), y_train)
            attack_indices = np.flatnonzero(held_mask)
            test_indices = np.concatenate([normal_test, attack_indices])
            rng = np.random.default_rng(seed)
            rng.shuffle(test_indices)
            y_test = held_mask[test_indices].astype(np.int8)
            probability = estimator.predict_proba(scaler.transform(X[test_indices]))[:, 1]
            errors = reconstruction_errors(autoencoder, scaler.transform(X[test_indices]))
            anomaly = anomaly_probability(errors, reference)
            classifier_prediction = (probability >= 0.5).astype(np.int8)
            hybrid_prediction, hybrid_score, route = fixed_router(
                probability, anomaly, 0.90, 0.99
            )
            for variant, prediction, score in (
                ("classifier_only", classifier_prediction, probability),
                ("unknown_attack_hybrid", hybrid_prediction, hybrid_score),
            ):
                seed_rows.append(
                    {
                        "seed": seed,
                        "held_out_attack_family": held_out,
                        "variant": variant,
                        **binary_metrics(y_test, prediction, score),
                        "classifier_route_fraction": (
                            1.0 if variant == "classifier_only" else float(route.mean())
                        ),
                    }
                )
        rows.extend(seed_rows)
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
    results = pd.DataFrame(rows)
    results.to_csv(checkpoint, index=False)
    summary = (
        results.groupby(["held_out_attack_family", "variant"])[
            ["recall", "f1", "mcc", "pr_auc", "fpr"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(resolve("archive/benchmark_v1/tables/unknown_attack_summary.csv"), index=False)
    return results
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/config.yaml")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    build_family_dataset()
    if not args.build_only:
        run(args.config)
if __name__ == "__main__":
    main()
