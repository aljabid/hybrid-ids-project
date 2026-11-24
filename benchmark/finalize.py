from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
from benchmark.common import environment_manifest, resolve, write_json
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
def run() -> None:
    write_json(
        resolve("artifacts/manifests/environment.json"), environment_manifest()
    )
    rows = []
    for path in sorted(resolve("artifacts").rglob("*")):
        if not path.is_file() or path.name == "artifact_inventory.csv":
            continue
        rows.append(
            {
                "path": str(path.relative_to(resolve("."))),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(
        resolve("artifacts/manifests/artifact_inventory.csv"), index=False
    )
if __name__ == "__main__":
    run()
