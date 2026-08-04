                                                                                

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmark" / "config.yaml"


def load_config(path: str | os.PathLike[str] = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for path in config["outputs"].values():
        resolve(path).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def row_hashes(values: np.ndarray) -> np.ndarray:
                                                                        
    normalized = np.ascontiguousarray(values, dtype=np.float64)
    return np.asarray(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in normalized]
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, default=str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def environment_manifest() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in (
        "numpy", "pandas", "sklearn", "scipy", "xgboost", "lightgbm",
        "catboost", "torch", "shap", "matplotlib", "seaborn", "psutil",
    ):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }

