                                                                 

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def model_and_space(name: str, seed: int, positive_weight: float) -> tuple[Any, dict]:
    if name == "decision_tree":
        model = DecisionTreeClassifier(
            random_state=seed, class_weight="balanced"
        )
        space = {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [4, 6, 8, 10, 15, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
        }
    elif name == "random_forest":
        model = RandomForestClassifier(
            random_state=seed, class_weight="balanced_subsample", n_jobs=1
        )
        space = {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [8, 12, 20, None],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2", 0.7],
        }
    elif name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            random_state=seed,
            n_jobs=1,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=positive_weight,
        )
        space = {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            verbosity=-1,
        )
        space = {
            "model__n_estimators": [50, 100, 150],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 6, 10, 15],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
        }
    elif name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            random_seed=seed,
            verbose=False,
            thread_count=1,
            auto_class_weights="Balanced",
            allow_writing_files=False,
        )
        space = {
            "model__iterations": [50, 100, 150],
            "model__depth": [4, 6, 8, 10],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__l2_leaf_reg": [1, 3, 5, 9],
        }
    elif name == "mlp":
        model = MLPClassifier(
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=250,
        )
        space = {
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.0005, 0.001, 0.005],
        }
    else:
        raise KeyError(f"Unknown model: {name}")
    return model, space


MODEL_NAMES = [
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
]
