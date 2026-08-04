                                                                            

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from benchmark.common import resolve, set_seed, write_json
from benchmark.model_registry import MODEL_NAMES, model_and_space


def config() -> dict:
    with resolve("benchmark/readiness_config.yaml").open() as handle:
        return yaml.safe_load(handle)


def paths() -> dict:
    result = {key: resolve(value) for key, value in config()["paths"].items()}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def load_data(partition: str):
    data = np.load(
        resolve(
            f"artifacts/readiness_v2/data/cicids2017_multifamily_{partition}.npz"
        ),
        allow_pickle=False,
    )
    return data["X"], data["y"], data["family"].astype(str), data["features"].astype(str).tolist()


def stratified_tuning_sample(
    X: np.ndarray, y: np.ndarray, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per_class = max_rows // 2
    selected = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(y == label),
                min(per_class, int((y == label).sum())),
                replace=False,
            )
            for label in (0, 1)
        ]
    )
    rng.shuffle(selected)
    return X[selected], y[selected]


def tune(force: bool = False) -> pd.DataFrame:
    output = paths()
    X, y, _, features = load_data("train")
    seed = 42
    X_tune, y_tune = stratified_tuning_sample(X, y, 100000, seed)
    positive_weight = float((y_tune == 0).sum() / (y_tune == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scorer = make_scorer(matthews_corrcoef)
    rows = []
    for name in MODEL_NAMES:
        result_path = output["results"] / f"tuning_{name}.json"
        model_path = output["models"] / f"{name}_tuned.pkl"
        if result_path.exists() and model_path.exists() and not force:
            rows.append(json.loads(result_path.read_text()))
            continue
        model, space = model_and_space(name, seed, positive_weight)
        pipeline = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=space,
            n_iter=5,
            scoring=scorer,
            cv=cv,
            random_state=seed,
            n_jobs=1,
            verbose=1,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(X_tune, y_tune)
        elapsed = time.perf_counter() - started
        with model_path.open("wb") as handle:
            pickle.dump(search.best_estimator_, handle)
        row = {
            "model": name,
            "best_cv_mcc": float(search.best_score_),
            "cv_std_at_best": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "best_parameters": search.best_params_,
            "search_seconds": elapsed,
            "tuning_rows": len(y_tune),
            "training_domain": "CICIDS2017 all available attack families",
        }
        write_json(result_path, row)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output["tables"] / "supervised_tuning_summary.csv", index=False)
    return table


def fit_random_forest_seed(seed: int = 42, force: bool = False):
    output = paths()
    model_path = output["models"] / f"random_forest_seed_{seed}.pkl"
    if model_path.exists() and not force:
        with model_path.open("rb") as handle:
            return pickle.load(handle)
    tuning = json.loads(
        (output["results"] / "tuning_random_forest.json").read_text()
    )
    X, y, _, _ = load_data("train")
    positive_weight = float((y == 0).sum() / (y == 1).sum())
    model, _ = model_and_space("random_forest", seed, positive_weight)
    pipeline = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
    pipeline.set_params(**tuning["best_parameters"])
    pipeline.fit(X, y)
    with model_path.open("wb") as handle:
        pickle.dump(pipeline, handle)
    return pipeline


if __name__ == "__main__":
    print(tune().to_string(index=False))
    fit_random_forest_seed(42)
