# Reproducible Hybrid IDS Benchmark

This directory implements the leakage-safe experimental protocol used for the
new benchmark. It does not overwrite the legacy `data/splits/local` directory
or the original saved results.

## Environment

```bash
python -m venv --system-site-packages .benchmark-venv
.benchmark-venv/bin/python -m pip install -r requirements-benchmark.txt
```

## One-command execution

```bash
MPLCONFIGDIR=/tmp/mpl-hybrid \
  .benchmark-venv/bin/python run_benchmark.py --phase all
```

Individual resumable stages:

```bash
.benchmark-venv/bin/python run_benchmark.py --phase data
.benchmark-venv/bin/python run_benchmark.py --phase tune
.benchmark-venv/bin/python run_benchmark.py --phase supervised
.benchmark-venv/bin/python run_benchmark.py --phase hybrid
.benchmark-venv/bin/python run_benchmark.py --phase unknown
.benchmark-venv/bin/python run_benchmark.py --phase analysis
```

## Protocol

- CICIDS2017 is split into disjoint training, validation, and internal-test
  partitions after exact duplicate and contradictory-row removal.
- CSE-CIC-IDS2018 is external-test only.
- Feature ordering and binary label semantics are identical across domains.
- Tuning uses training-domain cross-validation only.
- All final evaluations use ten paired seeds.
- Every seed-level prediction is saved for reproducible statistical testing.
- UNSW-NB15 is excluded until its source data and a defensible semantic feature
  mapping are available; see `artifacts/manifests/third_dataset_assessment.md`.

## Outputs

- `artifacts/data`: immutable compressed partitions
- `artifacts/models`: tuned and seed-specific estimators
- `artifacts/results`: raw seed and prediction results
- `artifacts/tables`: CSV and Excel tables
- `artifacts/figures`: IEEE-ready PDF and 600-DPI PNG figures
- `artifacts/manifests`: data, environment, and exclusion manifests

Official experimental runs must not use the legacy synthetic PCAP fallback.
