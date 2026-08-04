                                                                             

from __future__ import annotations

import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from benchmark.common import write_json
from benchmark.redesign_preprocessing import load_partition, paths
from benchmark.redesign_supervised import TabularResNet


TREE_MODELS = {
    "decision_tree": "group_tuned_decision_tree.pkl",
    "random_forest": "group_tuned_random_forest.pkl",
    "xgboost": "group_tuned_xgboost.pkl",
    "lightgbm": "group_tuned_lightgbm.pkl",
    "catboost": "group_tuned_catboost.pkl",
    "family_aware_rf": "objective_multiclass_family_balanced.pkl",
}
ALL_MODELS = list(TREE_MODELS) + ["redesigned_meta_hybrid", "tabular_resnet"]


def domain_sample(domain: str, seed=42, rows=200):
    output = paths()
    rng = np.random.default_rng(seed)
    if domain == "cicids2017_group_test":
        X, _, _, _, features = load_partition("internal_test")
    elif domain in ("cse_cic_ids2018", "cic_unsw_nb15"):
        with np.load(
            output["data"] / f"{domain}_expanded_diagnostic.npz",
            allow_pickle=False,
        ) as data:
            X = data["X"]
            features = data["features"].astype(str).tolist()
    elif domain == "cic_ton_iot_confirmation":
        metadata = json.loads(
            (
                output["data"]
                / "cic_ton_iot_confirmation_aligned/metadata.json"
            ).read_text()
        )
        X = np.load(
            output["data"] / "cic_ton_iot_confirmation_aligned/X.npy",
            mmap_mode="r",
        )[:metadata["valid_rows"]]
        features = metadata["feature_order"]
    else:
        raise KeyError(domain)
    index = rng.choice(len(X), min(rows, len(X)), replace=False)
    return np.asarray(X[index]), features


def feature_importance(values, feature_count):
    if isinstance(values, list):
        values = np.asarray(values)
    else:
        values = np.asarray(values)
    feature_axes = [axis for axis, size in enumerate(values.shape) if size == feature_count]
    if not feature_axes:
        raise ValueError(f"Cannot identify feature axis in SHAP shape {values.shape}")
    feature_axis = feature_axes[-1] if values.shape[-1] == feature_count else feature_axes[0]
    values = np.moveaxis(values, feature_axis, -1)
    return np.mean(np.abs(values), axis=tuple(range(values.ndim - 1)))


def explain_tree(model_name: str):
    output = paths()
    target = output["tables"] / f"shap_{model_name}.csv"
    if target.exists():
        return pd.read_csv(target)
    if model_name == "redesigned_meta_hybrid":
        with (output["models"] / "redesigned_hybrid_seed_42.pkl").open("rb") as handle:
            pipeline = pickle.load(handle)["classifier"]
        role = "hybrid supervised flow-feature backbone"
    else:
        with (output["models"] / TREE_MODELS[model_name]).open("rb") as handle:
            pipeline = pickle.load(handle)
        role = "model"
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    rows = []
    sample_rows = 50 if model_name in (
        "family_aware_rf", "redesigned_meta_hybrid"
    ) else 200
    for domain in (
        "cicids2017_group_test", "cse_cic_ids2018",
        "cic_unsw_nb15", "cic_ton_iot_confirmation",
    ):
        X, features = domain_sample(domain, rows=sample_rows)
        transformed = preprocessor.transform(X)
        values = explainer.shap_values(transformed)
        importance = feature_importance(values, len(features))
        for feature, value in zip(features, importance):
            rows.append({
                "model": model_name, "explanation_role": role,
                "dataset": domain, "feature": feature,
                "mean_abs_shap": float(value),
            })
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("dataset")["mean_abs_shap"].rank(
        ascending=False, method="first"
    ).astype(int)
    table.to_csv(target, index=False)
    return table


class ProbabilityWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values):
        return torch.sigmoid(self.model(values)).unsqueeze(1)


def explain_resnet():
    output = paths()
    target = output["tables"] / "shap_tabular_resnet.csv"
    if target.exists():
        return pd.read_csv(target)
    checkpoint = torch.load(
        output["models"] / "tabular_resnet_seed_42.pt",
        map_location="cpu", weights_only=False,
    )
    model = TabularResNet(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with (
        output["models"] / "tabular_resnet_preprocessor_seed_42.pkl"
    ).open("rb") as handle:
        preprocessor = pickle.load(handle)
    background_X, features = domain_sample("cicids2017_group_test", rows=40)
    wrapper = ProbabilityWrapper(model).eval()
    background = torch.as_tensor(preprocessor.transform(background_X))
    explainer = shap.DeepExplainer(wrapper, background)
    rows = []
    for domain in (
        "cicids2017_group_test", "cse_cic_ids2018",
        "cic_unsw_nb15", "cic_ton_iot_confirmation",
    ):
        X, _ = domain_sample(domain, rows=80)
        transformed = torch.as_tensor(preprocessor.transform(X))
        values = explainer.shap_values(transformed, check_additivity=False)
        importance = feature_importance(values, len(features))
        for feature, value in zip(features, importance):
            rows.append({
                "model": "tabular_resnet", "explanation_role": "model",
                "dataset": domain, "feature": feature,
                "mean_abs_shap": float(value),
            })
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("dataset")["mean_abs_shap"].rank(
        ascending=False, method="first"
    ).astype(int)
    table.to_csv(target, index=False)
    return table


def finalize():
    output = paths()
    tables = []
    missing = []
    for model in ALL_MODELS:
        target = output["tables"] / f"shap_{model}.csv"
        if target.exists():
            tables.append(pd.read_csv(target))
        else:
            missing.append(model)
    if missing:
        raise RuntimeError(f"Missing SHAP jobs: {missing}")
    table = pd.concat(tables, ignore_index=True)
    table.sort_values(["model", "dataset", "rank"]).to_csv(
        output["tables"] / "shap_feature_ranking.csv", index=False
    )
    table[table["rank"] <= 10].to_csv(
        output["tables"] / "shap_top10_features.csv", index=False
    )
    comparison = table.pivot_table(
        index=["model", "feature"], columns="dataset",
        values="mean_abs_shap",
    ).reset_index()
    comparison.to_csv(
        output["tables"] / "cross_dataset_shap_comparison.csv", index=False
    )
                                                            
    averaged = table.groupby(["model", "feature"], as_index=False)[
        "mean_abs_shap"
    ].mean()
    averaged["rank"] = averaged.groupby("model")["mean_abs_shap"].rank(
        ascending=False, method="first"
    )
    top = averaged[averaged["rank"] <= 10]
    fig, axes = plt.subplots(4, 2, figsize=(7.16, 10))
    for axis, model in zip(axes.ravel(), ALL_MODELS):
        part = top[top["model"] == model].sort_values("mean_abs_shap")
        axis.barh(part["feature"], part["mean_abs_shap"])
        axis.set_title(model.replace("_", " "))
        axis.set_xlabel("Cross-domain mean |SHAP|")
        axis.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(output["figures"] / "global_shap_summary.png", dpi=600)
    plt.close(fig)
    write_json(output["manifests"] / "explainability_scope.json", {
        "models": ALL_MODELS, "domains": 4,
        "tree_rows_per_domain": 200, "deep_rows_per_domain": 80,
        "real_rows_only": True,
        "hybrid_flow_feature_explanation": (
            "TreeSHAP of the hybrid supervised classifier backbone; the anomaly "
            "and meta-gate operate on derived scores and are not misrepresented "
            "as direct flow-feature SHAP values."
        ),
    })
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS + ["finalize"], required=True)
    arguments = parser.parse_args()
    if arguments.model == "finalize":
        print(finalize().groupby("model").size())
    elif arguments.model == "tabular_resnet":
        print(explain_resnet().groupby("dataset").size())
    else:
        print(explain_tree(arguments.model).groupby("dataset").size())


if __name__ == "__main__":
    main()
