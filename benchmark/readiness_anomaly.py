                                                                              

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler,
)
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark.common import resolve, set_seed, write_json
from benchmark.readiness_supervised import load_data


class ClippedMinMaxScaler(MinMaxScaler):
    def transform(self, X):
        return np.clip(super().transform(X), 0.0, 1.0)


def make_scaler(name: str, seed: int):
    if name == "minmax":
        return MinMaxScaler()
    if name == "clipped_minmax":
        return ClippedMinMaxScaler()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler(quantile_range=(5, 95))
    if name == "quantile":
        return QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            subsample=100000,
            random_state=seed,
        )
    raise KeyError(name)


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, X):
        return self.decoder(self.encoder(X))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.mu = nn.Linear(32, 8)
        self.logvar = nn.Linear(32, 8)
        self.decoder = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, input_dim)
        )

    def forward(self, X):
        hidden = self.hidden(X)
        mu, logvar = self.mu(hidden), self.logvar(hidden)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decoder(z), mu, logvar


class SVDDNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, X):
        return self.network(X)


@dataclass
class NeuralAnomaly:
    name: str
    model: nn.Module
    center: np.ndarray | None = None

    def score(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        values = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32)
                if self.name == "autoencoder":
                    reconstructed = self.model(batch)
                    score = torch.mean((batch - reconstructed) ** 2, dim=1)
                elif self.name == "variational_autoencoder":
                    reconstructed, _, _ = self.model(batch)
                    score = torch.mean((batch - reconstructed) ** 2, dim=1)
                elif self.name == "deep_svdd":
                    embedding = self.model(batch)
                    center = torch.as_tensor(self.center, dtype=torch.float32)
                    score = torch.sum((embedding - center) ** 2, dim=1)
                else:
                    raise KeyError(self.name)
                values.append(score.cpu().numpy())
        return np.concatenate(values)


@dataclass
class SklearnAnomaly:
    name: str
    model: object

    def score(self, X: np.ndarray) -> np.ndarray:
        return -np.asarray(self.model.decision_function(X), dtype=float)


def _loader(X: np.ndarray, seed: int, batch_size: int = 256):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def train_anomaly(name: str, X_normal: np.ndarray, seed: int, epochs: int = 20):
    set_seed(seed)
    if name == "isolation_forest":
        model = IsolationForest(
            n_estimators=250,
            max_samples=min(10000, len(X_normal)),
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        ).fit(X_normal)
        return SklearnAnomaly(name, model)
    if name == "one_class_svm":
        rng = np.random.default_rng(seed)
        subset = X_normal[
            rng.choice(len(X_normal), min(3000, len(X_normal)), replace=False)
        ]
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02).fit(subset)
        return SklearnAnomaly(name, model)
    if name == "local_outlier_factor":
        rng = np.random.default_rng(seed)
        subset = X_normal[
            rng.choice(len(X_normal), min(5000, len(X_normal)), replace=False)
        ]
        model = LocalOutlierFactor(
            n_neighbors=35, novelty=True, contamination="auto", n_jobs=1
        ).fit(subset)
        return SklearnAnomaly(name, model)
    loader = _loader(X_normal, seed)
    if name == "autoencoder":
        model = DenseAutoencoder(X_normal.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            model.train()
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = torch.mean((model(batch) - batch) ** 2)
                loss.backward()
                optimizer.step()
        return NeuralAnomaly(name, model)
    if name in {"denoising_autoencoder", "robust_autoencoder"}:
        model = DenseAutoencoder(X_normal.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        huber = nn.SmoothL1Loss()
        for _ in range(epochs):
            model.train()
            for (batch,) in loader:
                optimizer.zero_grad()
                model_input = (
                    batch + 0.02 * torch.randn_like(batch)
                    if name == "denoising_autoencoder"
                    else batch
                )
                reconstructed = model(model_input)
                loss = (
                    huber(reconstructed, batch)
                    if name == "robust_autoencoder"
                    else torch.mean((reconstructed - batch) ** 2)
                )
                loss.backward()
                optimizer.step()
                                                                       
        return NeuralAnomaly("autoencoder", model)
    if name == "variational_autoencoder":
        model = VariationalAutoencoder(X_normal.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            model.train()
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstructed, mu, logvar = model(batch)
                reconstruction = torch.mean((reconstructed - batch) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction + 1e-3 * kl
                loss.backward()
                optimizer.step()
        return NeuralAnomaly(name, model)
    if name == "deep_svdd":
        model = SVDDNetwork(X_normal.shape[1])
        model.eval()
        with torch.no_grad():
            center = np.mean(
                np.concatenate(
                    [
                        model(batch).numpy()
                        for (batch,) in DataLoader(
                            TensorDataset(torch.as_tensor(X_normal, dtype=torch.float32)),
                            batch_size=4096,
                        )
                    ]
                ),
                axis=0,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        center_tensor = torch.as_tensor(center, dtype=torch.float32)
        for _ in range(epochs):
            model.train()
            for (batch,) in loader:
                optimizer.zero_grad()
                embedding = model(batch)
                loss = torch.mean(torch.sum((embedding - center_tensor) ** 2, dim=1))
                loss.backward()
                optimizer.step()
        return NeuralAnomaly(name, model, center=center)
    raise KeyError(name)


def empirical_probability(scores: np.ndarray, normal_reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(normal_reference))
    return np.searchsorted(ordered, scores, side="right") / len(ordered)


def select_threshold(
    y: np.ndarray, probability: np.ndarray, quantiles: list[float]
) -> tuple[float, float]:
    best_threshold, best_mcc = quantiles[0], -np.inf
    for threshold in quantiles:
        mcc = matthews_corrcoef(y, probability >= threshold)
        if mcc > best_mcc:
            best_threshold, best_mcc = threshold, float(mcc)
    return float(best_threshold), float(best_mcc)


def pilot_selection(seed: int = 42) -> pd.DataFrame:
    config = yaml_safe_load()
    root = resolve(config["paths"]["root"])
    for value in config["paths"].values():
        resolve(value).mkdir(parents=True, exist_ok=True)
    X_train, y_train, _, _ = load_data("train")
    X_validation, y_validation, _, _ = load_data("validation")
    rng = np.random.default_rng(seed)
                                                                               
                                                                          
                                                                           
    validation_cap = int(config["selection"].get("validation_selection_cap", 100000))
    per_class = validation_cap // 2
    validation_indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(y_validation == label),
                min(per_class, int((y_validation == label).sum())),
                replace=False,
            )
            for label in (0, 1)
        ]
    )
    rng.shuffle(validation_indices)
    X_validation = X_validation[validation_indices]
    y_validation = y_validation[validation_indices]
    normal_train = X_train[y_train == 0]
    cap = int(config["selection"]["anomaly_training_normal_cap"])
    normal_fit = normal_train[
        rng.choice(len(normal_train), min(cap, len(normal_train)), replace=False)
    ]
    rows = []
    for scaler_name in config["selection"]["scaler_candidates"]:
        scaler = make_scaler(scaler_name, seed).fit(normal_fit)
        train_scaled = scaler.transform(normal_fit).astype(np.float32)
        validation_scaled = scaler.transform(X_validation).astype(np.float32)
        normal_validation = validation_scaled[y_validation == 0]
        for model_name in config["anomaly_models"]:
            started = time.perf_counter()
            detector = train_anomaly(model_name, train_scaled, seed)
            training_seconds = time.perf_counter() - started
            reference_score = detector.score(normal_validation)
            score = detector.score(validation_scaled)
            probability = empirical_probability(score, reference_score)
            threshold, mcc = select_threshold(
                y_validation,
                probability,
                config["selection"]["validation_threshold_quantiles"],
            )
            rows.append(
                {
                    "seed": seed,
                    "scaler": scaler_name,
                    "anomaly_model": model_name,
                    "validation_mcc": mcc,
                    "validation_pr_auc": float(
                        average_precision_score(y_validation, probability)
                    ),
                    "validation_roc_auc": float(
                        roc_auc_score(y_validation, probability)
                    ),
                    "selected_threshold": threshold,
                    "training_seconds": training_seconds,
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["validation_mcc", "validation_pr_auc"], ascending=False
    )
    table.to_csv(
        resolve(config["paths"]["tables"]) / "anomaly_scaler_pilot.csv", index=False
    )
    winner = table.iloc[0].to_dict()
    write_json(
        resolve(config["paths"]["manifests"]) / "anomaly_scaler_selection.json",
        {
            "selection_source": (
                "fixed stratified sample of leakage-audited, multi-family "
                "CICIDS2017 validation partition only"
            ),
            "selection_rows": int(len(y_validation)),
            "ranking_rule": "validation MCC, tie-break PR-AUC",
            "winner": winner,
        },
    )
    return table


def yaml_safe_load() -> dict:
    import yaml

    with resolve("benchmark/readiness_config.yaml").open() as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    result = pilot_selection()
    print(result.head(10).to_string(index=False))
