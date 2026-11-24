from __future__ import annotations
import argparse
import json
import pickle
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from benchmark.readiness_anomaly import empirical_probability
from benchmark.redesign_preprocessing import load_partition, paths
from benchmark.redesign_supervised import TabularResNet
from benchmark.redesign_validation import _meta_features
MODEL_FILES = {
    "decision_tree": ["group_tuned_decision_tree.pkl"],
    "random_forest": ["group_tuned_random_forest.pkl"],
    "xgboost": ["group_tuned_xgboost.pkl"],
    "lightgbm": ["group_tuned_lightgbm.pkl"],
    "catboost": ["group_tuned_catboost.pkl"],
    "family_aware_rf": ["objective_multiclass_family_balanced.pkl"],
    "redesigned_meta_hybrid": ["redesigned_hybrid_seed_42.pkl"],
    "tabular_resnet": [
        "tabular_resnet_seed_42.pt", "tabular_resnet_preprocessor_seed_42.pkl",
    ],
}
def artifact_size(model):
    root = paths()["models"]
    return sum((root / name).stat().st_size for name in MODEL_FILES[model])
def run(model_name):
    output = paths()
    X, _, _, _, _ = load_partition("internal_test")
    index = np.random.default_rng(42).choice(len(X), 100_000, replace=False)
    X = X[index]
    model_root = output["models"]
    started = time.perf_counter()
    if model_name in (
        "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
    ):
        with (model_root / MODEL_FILES[model_name][0]).open("rb") as handle:
            model = pickle.load(handle)
        probability = model.predict_proba(X)[:, 1]
    elif model_name == "family_aware_rf":
        with (model_root / MODEL_FILES[model_name][0]).open("rb") as handle:
            model = pickle.load(handle)
        raw = model.predict_proba(X)
        benign = int(np.min(model.classes_))
        probability = 1 - raw[:, list(model.classes_).index(benign)]
    elif model_name == "redesigned_meta_hybrid":
        with (model_root / MODEL_FILES[model_name][0]).open("rb") as handle:
            bundle = pickle.load(handle)
        classifier = bundle["classifier"]
        raw = classifier.predict_proba(X)
        benign = int(np.min(classifier.classes_))
        classifier_probability = bundle["calibrator"].transform(
            1 - raw[:, list(classifier.classes_).index(benign)]
        )
        anomaly_probability = empirical_probability(
            bundle["anomaly_detector"].score(
                bundle["anomaly_scaler"].transform(X).astype(np.float32)
            ),
            bundle["anomaly_reference"],
        )
        probability = bundle["meta_model"].predict_proba(
            _meta_features(classifier_probability, anomaly_probability)
        )[:, 1]
    else:
        checkpoint = torch.load(
            model_root / "tabular_resnet_seed_42.pt",
            map_location="cpu", weights_only=False,
        )
        network = TabularResNet(checkpoint["input_dim"])
        network.load_state_dict(checkpoint["state_dict"]); network.eval()
        with (model_root / "tabular_resnet_preprocessor_seed_42.pkl").open("rb") as handle:
            preprocessor = pickle.load(handle)
        with torch.no_grad():
            probability = torch.sigmoid(
                network(torch.as_tensor(preprocessor.transform(X)))
            ).numpy()
    elapsed = time.perf_counter() - started
    row = {
        "model": model_name, "measurement_rows": len(X),
        "inference_seconds_100k": elapsed,
        "inference_us_per_flow": elapsed / len(X) * 1e6,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "model_bytes": artifact_size(model_name),
        "probability_checksum": float(np.sum(probability, dtype=np.float64)),
        "measurement_scope": "isolated process; 100,000 real CICIDS2017 group-test flows",
    }
    target = output["tables"] / f"efficiency_{model_name}.json"
    target.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    return row
def finalize():
    output = paths()
    rows = [
        json.loads((output["tables"] / f"efficiency_{model}.json").read_text())
        for model in MODEL_FILES
    ]
    measured = pd.DataFrame(rows)
    timing = pd.read_csv(output["tables"] / "efficiency_results.csv")
    final = timing.merge(measured, on="model", how="left")
    final.to_csv(output["tables"] / "computational_efficiency.csv", index=False)
    return final
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_FILES) + ["finalize"], required=True)
    arguments = parser.parse_args()
    if arguments.model == "finalize":
        print(finalize().to_string(index=False))
    else:
        run(arguments.model)
if __name__ == "__main__":
    main()
