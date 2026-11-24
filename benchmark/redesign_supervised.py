from __future__ import annotations
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from benchmark.common import resolve, set_seed, write_json
from benchmark.metrics import binary_metrics
from benchmark.redesign_preprocessing import (
    RobustFlowTransformer, config, load_partition, paths, stratified_sample,
)
MODEL_NAMES = ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
def model_and_space(name, seed):
    if name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced", random_state=seed
        ), {
            "model__max_depth": [8, 12, 18, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__criterion": ["gini", "entropy", "log_loss"],
        }
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample", random_state=seed, n_jobs=1
        ), {
            "model__n_estimators": [40, 60, 80],
            "model__max_depth": [12, 18],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", 0.5],
        }
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss"
        ), {
            "model__n_estimators": [50, 75, 100],
            "model__max_depth": [4, 6, 9],
            "model__learning_rate": [0.03, 0.07, 0.15],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
        }
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            random_state=seed, n_jobs=1, class_weight="balanced", verbosity=-1
        ), {
            "model__n_estimators": [50, 75, 100],
            "model__num_leaves": [31, 63, 127],
            "model__learning_rate": [0.03, 0.07, 0.15],
            "model__max_depth": [-1, 8, 14],
        }
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            random_seed=seed, verbose=False, thread_count=1,
            auto_class_weights="Balanced", allow_writing_files=False,
        ), {
            "model__iterations": [50, 75, 100],
            "model__depth": [5, 7, 9],
            "model__learning_rate": [0.03, 0.07, 0.15],
            "model__l2_leaf_reg": [1, 3, 7],
        }
    raise KeyError(name)
def selected_transformer(seed=42):
    X, _, _, _, features = load_partition("train")
    log_indices = tuple(
        features.index(name) for name in config()["data"]["log1p_features"]
        if name in features
    )
    return RobustFlowTransformer(
        method="robust", log_indices=log_indices, random_state=seed
    )
def tune(seed=42, force=False):
    output = paths()
    X, y, _, groups, _ = load_partition("train")
    index = stratified_sample(y, config()["selection"]["tuning_rows"], seed)
    Xs, ys, gs = X[index], y[index], groups[index]
    cv = GroupKFold(n_splits=len(np.unique(gs)))
    scorer = make_scorer(matthews_corrcoef)
    rows = []
    for name in MODEL_NAMES:
        target = output["results"] / f"group_tuning_{name}.json"
        model_target = output["models"] / f"group_tuned_{name}.pkl"
        if target.exists() and model_target.exists() and not force:
            rows.append(json.loads(target.read_text()))
            continue
        estimator, space = model_and_space(name, seed)
        pipeline = Pipeline([
            ("preprocessor", selected_transformer(seed)),
            ("model", estimator),
        ])
        search = RandomizedSearchCV(
            pipeline, space, n_iter=3, scoring=scorer, cv=cv,
            random_state=seed, n_jobs=1, error_score="raise", verbose=1,
        )
        started = time.perf_counter()
        search.fit(Xs, ys, groups=gs)
        row = {
            "model": name, "best_group_cv_mcc": float(search.best_score_),
            "group_cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_parameters": search.best_params_,
            "search_seconds": time.perf_counter() - started,
            "rows": len(ys), "groups": sorted(np.unique(gs).tolist()),
        }
        with model_target.open("wb") as handle:
            pickle.dump(search.best_estimator_, handle)
        write_json(target, row)
        rows.append(row)
        print(f"tuned {name}: group-CV MCC={row['best_group_cv_mcc']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "group_tuning_summary.csv", index=False)
    return table
def family_balanced_indices(family, y, seed, benign_cap=100000, per_family=20000):
    rng = np.random.default_rng(seed)
    selected = []
    for value in np.unique(family):
        indices = np.flatnonzero(family == value)
        cap = benign_cap if y[indices[0]] == 0 else per_family
        selected.append(rng.choice(indices, min(cap, len(indices)), replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result
def compare_objectives(seed=42, force=False):
    output = paths()
    target = output["tables"] / "family_aware_objective_selection.csv"
    if target.exists() and not force:
        return pd.read_csv(target)
    X, y, family, _, _ = load_partition("train")
    Xv, yv, _, _, _ = load_partition("validation")
    validation = stratified_sample(yv, config()["selection"]["validation_rows"], seed)
    tuning = json.loads(
        (output["results"] / "group_tuning_random_forest.json").read_text()
    )
    index = family_balanced_indices(family, y, seed)
    rows = []
    for objective, target_y in (
        ("binary_balanced", y),
        ("multiclass_family_balanced", family),
    ):
        estimator, _ = model_and_space("random_forest", seed)
        pipeline = Pipeline([
            ("preprocessor", selected_transformer(seed)), ("model", estimator)
        ])
        pipeline.set_params(**tuning["best_parameters"])
        pipeline.fit(X[index], target_y[index])
        probabilities = pipeline.predict_proba(Xv[validation])
        if objective == "binary_balanced":
            attack_probability = probabilities[:, list(pipeline.classes_).index(1)]
        else:
            benign_class = int(np.unique(family[y == 0])[0])
            attack_probability = 1 - probabilities[:, list(pipeline.classes_).index(benign_class)]
        metrics = binary_metrics(
            yv[validation], attack_probability >= 0.5, attack_probability
        )
        rows.append({
            "objective": objective, "training_rows": len(index),
            "robust_score": metrics["mcc"] - 0.5 * metrics["fpr"], **metrics,
        })
        with (output["models"] / f"objective_{objective}.pkl").open("wb") as handle:
            pickle.dump(pipeline, handle)
    table = pd.DataFrame(rows).sort_values("robust_score", ascending=False)
    table.to_csv(target, index=False)
    write_json(output["manifests"] / "supervised_objective_selection.json", {
        "selection_data": "held-out CICIDS2017 validation groups",
        "external_labels_used": False, "winner": table.iloc[0].to_dict(),
    })
    return table
class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(width, width), nn.BatchNorm1d(width),
        )
        self.activation = nn.ReLU()
    def forward(self, X):
        return self.activation(X + self.block(X))
class TabularResNet(nn.Module):
    def __init__(self, input_dim, width=128, blocks=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.ReLU(),
            *[ResidualBlock(width) for _ in range(blocks)],
            nn.Linear(width, 1),
        )
    def forward(self, X):
        return self.network(X).squeeze(1)
def train_deep(seed=42, force=False):
    output = paths()
    target = output["models"] / f"tabular_resnet_seed_{seed}.pt"
    history_target = output["tables"] / f"tabular_resnet_history_seed_{seed}.csv"
    if target.exists() and history_target.exists() and not force:
        return pd.read_csv(history_target)
    set_seed(seed)
    X, y, family, _, _ = load_partition("train")
    Xv, yv, _, _, _ = load_partition("validation")
    rng = np.random.default_rng(seed)
    train_index = family_balanced_indices(
        family, y, seed, benign_cap=150000, per_family=30000
    )
    cap = config()["selection"]["deep_training_rows"]
    if len(train_index) > cap:
        train_index = rng.choice(train_index, cap, replace=False)
    val_index = stratified_sample(yv, config()["selection"]["validation_rows"], seed)
    preprocessor = selected_transformer(seed).fit(X[train_index])
    Xt = preprocessor.transform(X[train_index])
    Xval = preprocessor.transform(Xv[val_index])
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(Xt), torch.as_tensor(y[train_index], dtype=torch.float32)
        ),
        batch_size=2048, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TabularResNet(X.shape[1])
    positive_weight = torch.tensor(
        [(y[train_index] == 0).sum() / max((y[train_index] == 1).sum(), 1)],
        dtype=torch.float32,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    Xval_tensor = torch.as_tensor(Xval)
    best_mcc, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(config()["selection"]["deep_epochs"]):
        model.train()
        losses = []
        for batch, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(model(Xval_tensor)).numpy()
        metrics = binary_metrics(yv[val_index], probability >= 0.5, probability)
        scheduler.step(metrics["mcc"])
        history.append({
            "epoch": epoch + 1, "loss": np.mean(losses),
            "validation_mcc": metrics["mcc"], "validation_tpr": metrics["tpr"],
            "validation_fpr": metrics["fpr"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if metrics["mcc"] > best_mcc + 1e-4:
            best_mcc = metrics["mcc"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"resnet epoch={epoch+1} val_mcc={metrics['mcc']:.4f}", flush=True)
        if stale >= config()["selection"]["deep_patience"]:
            break
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(), "input_dim": X.shape[1],
        "seed": seed, "best_validation_mcc": best_mcc,
    }, target)
    with (output["models"] / f"tabular_resnet_preprocessor_seed_{seed}.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle)
    table = pd.DataFrame(history)
    table.to_csv(history_target, index=False)
    return table
if __name__ == "__main__":
    print(tune().to_string(index=False))
    print(compare_objectives().to_string(index=False))
    train_deep()
