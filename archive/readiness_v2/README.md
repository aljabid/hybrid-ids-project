# Manuscript-readiness experiment package (v2)

This directory contains the real-data, frozen-protocol evaluation of the Hybrid
IDS project. It does not contain a manuscript and does not modify the research
objective.

## Scientific status

The experiments are complete enough to support a **negative cross-domain
robustness study**, but they do **not** support a claim that the current hybrid
detector is superior. The frozen hybrid fails the predeclared cross-domain MCC
and TPR gates. See `manifests/final_readiness_verdict.json` and
`manifests/claims_audit.json`.

No mock, generated, or synthetic flows were used. External labels were not used
to select models, scaling, calibration, anomaly detection, thresholds, or
fusion.

## Real datasets

- CICIDS2017: training, validation, and internal test; all available attack
  families; feature-identical and contradictory duplicates removed before the
  split.
- CSE-CIC-IDS2018: deduplicated external test.
- CIC-UNSW-NB15: real CICFlowMeter re-extraction of UNSW-NB15, external test.

The supplied CICIDS2017 files lack timestamps/source identifiers, and the
CSE-CIC-IDS2018 extract lacks source identifiers. Consequently, the proposed
per-source temporal contribution was removed from validated claims.

## Reproduction order

Run from the repository root with `.benchmark-venv/bin/python`:

1. `python -m benchmark.readiness_data`
2. `python -m benchmark.readiness_supervised`
3. `python -m benchmark.readiness_anomaly`
4. `python -m benchmark.readiness_fusion`
5. `python -m benchmark.readiness_final`
6. `python -m benchmark.readiness_unknown_v2`
7. `python -m benchmark.readiness_anomaly_final`
8. `python -m benchmark.readiness_analysis_v2`
9. `python -m benchmark.readiness_finalize_v2`

All long stages checkpoint CSV results after each model/seed. Configuration and
success gates are in `benchmark/readiness_config.yaml`; the frozen scientific
rules are in `PROTOCOL.md`.

## Artifact map

- `data/`: processed real-data arrays
- `models/`: tuned and per-seed fitted models
- `results/`: raw seed-level metrics and seed-42 prediction archives
- `tables/`: CSV and Excel tables
- `figures/`: 600-DPI publication figures
- `manifests/`: dataset hashes, method freeze, claims audit, and final verdict
- `ARTIFACT_INVENTORY.csv`: size and SHA-256 of reportable artifacts

The MLP comparator uses a declared 20-iteration cap and batch size 2,048. It
reaches the cap and should be described as a resource-bounded, non-converged
baseline—not as a fully optimized deep-learning result.
