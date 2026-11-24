from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from benchmark.common import load_config, resolve, set_seed, write_json
from benchmark.metrics import binary_metrics
from benchmark.supervised import load_partition
class FlowAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )
    def forward(self, values):
        return self.network(values)
def reconstruction_errors(model: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32)
            reconstructed = model(batch)
            output.append(torch.mean((batch - reconstructed) ** 2, dim=1).cpu().numpy())
    return np.concatenate(output)
def train_autoencoder(
    X_normal: np.ndarray, seed: int, epochs: int = 30, batch_size: int = 256
) -> tuple[FlowAutoencoder, list[float], float]:
    set_seed(seed)
    model = FlowAutoencoder(X_normal.shape[1])
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_normal, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    history = []
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        losses = []
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history, time.perf_counter() - started
def _positive_probability(classifier, X: np.ndarray) -> np.ndarray:
    return np.asarray(classifier.predict_proba(X))[:, 1]
def anomaly_probability(errors: np.ndarray, normal_reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(normal_reference)
    return np.searchsorted(ordered, errors, side="right") / len(ordered)
def fixed_router(
    attack_probability: np.ndarray,
    anomaly_probability_values: np.ndarray,
    confidence_threshold: float,
    anomaly_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = attack_probability >= 0.5
    confidence = np.maximum(attack_probability, 1 - attack_probability)
    use_classifier = prediction & (confidence >= confidence_threshold)
    final = np.where(
        use_classifier,
        1,
        anomaly_probability_values >= anomaly_threshold,
    ).astype(np.int8)
    combined_score = np.where(
        use_classifier, attack_probability, anomaly_probability_values
    )
    return final, combined_score, use_classifier
def adaptive_router(
    attack_probability: np.ndarray,
    anomaly_probability_values: np.ndarray,
    initial_threshold: float,
    anomaly_threshold: float,
    window_size: int = 50,
    update_rate: float = 0.1,
    minimum: float = 0.70,
    maximum: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    final = np.zeros(len(attack_probability), dtype=np.int8)
    score = np.zeros(len(attack_probability), dtype=float)
    classifier_route = np.zeros(len(attack_probability), dtype=bool)
    thresholds = np.zeros(len(attack_probability), dtype=float)
    confidence_window: list[float] = []
    current = initial_threshold
    for index, probability in enumerate(attack_probability):
        confidence = max(probability, 1 - probability)
        thresholds[index] = current
        classifier_route[index] = probability >= 0.5 and confidence >= current
        if classifier_route[index]:
            final[index] = 1
            score[index] = probability
        else:
            final[index] = anomaly_probability_values[index] >= anomaly_threshold
            score[index] = anomaly_probability_values[index]
        confidence_window.append(confidence)
        if len(confidence_window) > window_size:
            confidence_window.pop(0)
        if len(confidence_window) == window_size:
            current = float(
                np.clip(
                    (1 - update_rate) * current
                    + update_rate * np.mean(confidence_window),
                    minimum,
                    maximum,
                )
            )
    return final, score, classifier_route, thresholds
def _select_backbone() -> str:
    paths = list(resolve("artifacts/results").glob("tuning_*.json"))
    if not paths:
        raise FileNotFoundError("No tuning results. Run supervised tuning first.")
    candidates = [json.loads(path.read_text()) for path in paths]
    return max(candidates, key=lambda row: row["best_cv_mcc"])["model"]
def run(config_path: str = "benchmark/config.yaml") -> pd.DataFrame:
    config = load_config(resolve(config_path))
    seeds = [int(value) for value in config["experiments"]["seeds"]]
    backbone = _select_backbone()
    X_train, y_train, _ = load_partition("cicids2017_train")
    X_validation, y_validation, _ = load_partition("cicids2017_validation")
    datasets = {
        "cicids2017_internal": load_partition("cicids2017_test")[:2],
        "cic_ids2018_external": load_partition("cic_ids2018_external")[:2],
    }
    rows = []
    prediction_dir = resolve("artifacts/results/hybrid_predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        classifier_path = resolve(f"artifacts/models/{backbone}_seed_{seed}.pkl")
        if not classifier_path.exists():
            raise FileNotFoundError(
                f"Backbone seed model missing: {classifier_path}. Run supervised benchmark."
            )
        with classifier_path.open("rb") as handle:
            classifier = pickle.load(handle)
        selector = classifier[:-1]
        X_train_scaled = selector.transform(X_train)
        X_validation_scaled = selector.transform(X_validation)
        autoencoder, history, training_seconds = train_autoencoder(
            X_train_scaled[y_train == 0], seed
        )
        normal_reference = reconstruction_errors(
            autoencoder, X_validation_scaled[y_validation == 0]
        )
        validation_errors = reconstruction_errors(autoencoder, X_validation_scaled)
        validation_anomaly = anomaly_probability(validation_errors, normal_reference)
        threshold_candidates = [0.95, 0.975, 0.99, 0.995]
        threshold_scores = {}
        for threshold in threshold_candidates:
            predicted = (validation_anomaly >= threshold).astype(np.int8)
            threshold_scores[str(threshold)] = binary_metrics(
                y_validation, predicted, validation_anomaly
            )["mcc"]
        anomaly_threshold = max(
            threshold_candidates, key=lambda value: threshold_scores[str(value)]
        )
        torch.save(
            {
                "state_dict": autoencoder.state_dict(),
                "input_dim": X_train_scaled.shape[1],
                "normal_reference": normal_reference,
                "anomaly_threshold": anomaly_threshold,
            },
            resolve(f"artifacts/models/autoencoder_seed_{seed}.pt"),
        )
        write_json(
            resolve(f"artifacts/results/autoencoder_seed_{seed}.json"),
            {
                "seed": seed,
                "backbone": backbone,
                "epochs": len(history),
                "final_loss": history[-1],
                "training_seconds": training_seconds,
                "threshold_mcc": threshold_scores,
                "selected_anomaly_percentile": anomaly_threshold,
            },
        )
        for dataset, (X, y) in datasets.items():
            X_scaled = selector.transform(X)
            probability = _positive_probability(classifier, X)
            errors = reconstruction_errors(autoencoder, X_scaled)
            anomaly = anomaly_probability(errors, normal_reference)
            variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
                "classifier_only": (
                    (probability >= 0.5).astype(np.int8),
                    probability,
                    np.ones(len(y), dtype=bool),
                ),
                "autoencoder_only": (
                    (anomaly >= anomaly_threshold).astype(np.int8),
                    anomaly,
                    np.zeros(len(y), dtype=bool),
                ),
                "fixed_hybrid": fixed_router(
                    probability, anomaly, 0.85, anomaly_threshold
                ),
                "confidence_hybrid": fixed_router(
                    probability, anomaly, 0.90, anomaly_threshold
                ),
            }
            adaptive_prediction, adaptive_score, adaptive_route, thresholds = adaptive_router(
                probability, anomaly, 0.85, anomaly_threshold
            )
            variants["adaptive_hybrid"] = (
                adaptive_prediction,
                adaptive_score,
                adaptive_route,
            )
            for variant, (prediction, combined_score, route) in variants.items():
                row = {
                    "variant": variant,
                    "backbone": backbone,
                    "seed": seed,
                    "dataset": dataset,
                    **binary_metrics(y, prediction, combined_score),
                    "classifier_route_fraction": float(route.mean()),
                    "autoencoder_route_fraction": float((~route).mean()),
                    "anomaly_threshold_percentile": anomaly_threshold,
                }
                rows.append(row)
                np.savez_compressed(
                    prediction_dir / f"{variant}_{dataset}_seed_{seed}.npz",
                    y_true=y.astype(np.int8),
                    y_pred=prediction,
                    score=combined_score.astype(np.float32),
                    attack_probability=probability.astype(np.float32),
                    anomaly_probability=anomaly.astype(np.float32),
                    reconstruction_error=errors.astype(np.float32),
                    classifier_route=route,
                    adaptive_threshold=(
                        thresholds.astype(np.float32)
                        if variant == "adaptive_hybrid"
                        else np.full(len(y), np.nan, dtype=np.float32)
                    ),
                )
    results = pd.DataFrame(rows)
    results.to_csv(resolve("artifacts/results/hybrid_seed_results.csv"), index=False)
    metrics = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "mcc", "tpr", "fpr", "ece", "brier",
        "negative_log_likelihood", "classifier_route_fraction",
        "autoencoder_route_fraction",
    ]
    summary = results.groupby(["variant", "dataset"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(resolve("artifacts/tables/hybrid_ablation_summary.csv"), index=False)
    return results
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/config.yaml")
    args = parser.parse_args()
    run(args.config)
if __name__ == "__main__":
    main()
