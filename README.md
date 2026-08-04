# Hybrid Network Intrusion Detection: A Group-Aware Evaluation

**A leakage-controlled, group-aware, multi-domain evaluation of machine-learning intrusion detection — including a hybrid detector whose promising internal and held-out-family results did *not* survive a pre-frozen, untouched external confirmation.**

This repository is the complete, reproducible evidence base for that finding: real network-flow data only, ten paired seeds, an evidence hierarchy that separates split-strategy effects from family exclusion from true cross-domain transfer, and one dataset that was never opened until the method was frozen.

> **The central result is a negative one, on purpose.** A matched random-row-versus-group-aware control shows internal benchmark scores are conditional on split strategy; held-out-family success did not reliably predict cross-domain performance; and the frozen hybrid detector reached MCC ≈ 0 on the untouched confirmation dataset. See [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md) for the full scientific status and the exact claims this evidence does and does not support.

---

## What's in this repository

| You want to... | Go to |
|---|---|
| **See the authoritative experimental results** | [`artifacts/redesign_v3/`](artifacts/redesign_v3/) — the current, real-data-only, group-aware experiment generation |
| **Reproduce the core benchmark** | [`benchmark/`](benchmark/) and [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md#reproduction-order) |
| **Understand the project's evolution** | See [Experiment generations](#experiment-generations) below — this project went through three, and only the third is authoritative |
| **Look at the original hybrid-IDS pipeline** | See [Legacy pipeline](#legacy-pipeline-historical) — superseded, kept for history |

---

## The question this project answers

> **Do favorable held-out-family and internal evaluation results reliably predict cross-domain intrusion-detection performance?**

The answer, for the datasets, models, and protocol evaluated here, is **no** — and the repository is structured to show *why*, not just assert it:

1. **A matched random-row vs. group-aware control** (all 8 benchmarked configurations, 10 seeds each) shows that split strategy alone changes internal MCC, with the strongest possible non-parametric signal at this sample size.
2. **A group-aware CICIDS2017 protocol** with zero cross-partition feature-hash overlap separates within-collection generalization from cross-collection transfer.
3. **Held-out-attack-family exclusion** shows omitted attack types can remain detectable within one collection without that success transferring to another collection.
4. **A shift-mechanism decomposition** (label-conditional drift, an oracle-threshold bound, post-hoc recalibration) shows *why* cross-domain scores collapse — and that the mechanism differs by domain even under one identical pipeline.
5. **A frozen, untouched external confirmation** on CIC-ToN-IoT (4,847,499 real flows, never accessed until the method was locked) is the strongest test in the package, and the hybrid detector failed it.

## Experiment generations

This project iterated through three experimental generations. Only the third is authoritative for any number quoted from this repository.

| Generation | Location | Status |
|---|---|---|
| 1. Legacy pipeline | `main.py`, `models/`, `hybrid/`, `baselines/`, `results/` | Historical only — see [below](#legacy-pipeline-historical) |
| 2. Leakage-audited benchmark | `benchmark/`, `artifacts/readiness_v2/` | Superseded by generation 3 |
| 3. **Group-aware, real-data-only redesign** | **[`artifacts/redesign_v3/`](artifacts/redesign_v3/)** | **Authoritative.** All manuscript numbers trace here. |

When two files in this repository appear to disagree, [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md) documents the exact evidence hierarchy to resolve it (manifests > final tables > raw run CSVs > source code > legacy files).

## Reproducing the current results

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-benchmark.txt   # Windows
# .venv/bin/pip install -r requirements-benchmark.txt     # Linux/macOS
.venv/Scripts/python -m pytest -q   # expect 55 passed
```

`requirements-benchmark.txt` pins exact versions, verified against the
environment that actually produced every result in this repository (see
`artifacts/redesign_v3/QA_REPORT.md` for the verification record — a
previous version of this file used version *ranges* that did not match the
tested environment; that mismatch has been fixed).

Real public datasets are required and are **not** included in this repository (see licensing note below): place CICIDS2017, CSE-CIC-IDS2018, CIC-UNSW-NB15, and the CIC-ToN-IoT V2 confirmation release at the paths documented in [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md#real-datasets), then follow that file's reproduction order. The matched-split control, shift-mechanism diagnostics, recalibration, and selective-classifier baseline are documented with their exact commands there as well. The robustness/ablation/second-confirmation addendum (not yet part of the manuscript; see `artifacts/redesign_v3/ROBUSTNESS_AND_ABLATION_ADDENDUM.md`) has its own reproduction commands in that file, including a Kaggle API token requirement for re-downloading the CIC-DDoS2019 second-confirmation dataset.

## Repository layout

```
hybrid_ids_project/
├── artifacts/redesign_v3/          # Authoritative experiment generation (data, tables, figures, manifests)
├── benchmark/                      # Generation-2/3 experiment source code (all Python entry points)
├── config/                         # Central configuration (legacy pipeline)
├── data/raw/                       # Where to place downloaded public datasets (not tracked in git)
├── requirements.txt                # Legacy pipeline dependencies
├── requirements-benchmark.txt      # Current benchmark dependencies
├── tests/                          # Automated tests (55 passing)
│
├── main.py, models/, hybrid/,      # Legacy pipeline — see "Legacy pipeline" below
│   baselines/, preprocessing/,
│   explainability/, calibration/,
│   validation/, results/, utils/
```

## Citation

If you use the data alignment, evaluation protocol, or findings from this repository, please cite the source datasets:

```
CICIDS2017:
Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new
intrusion detection dataset and intrusion traffic characterization. ICISSP 2018.

CSE-CIC-IDS2018:
Sharafaldin, I., et al. CSE-CIC-IDS2018. Communications Security Establishment
& University of New Brunswick, 2018.

UNSW-NB15:
Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network
intrusion detection systems. MilCIS 2015.

TON_IoT:
Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., & Anwar, A. (2020). TON_IoT
telemetry dataset. IEEE Access, 8, 165130-165150.
```

## Data availability

No raw or processed dataset rows are committed to this repository. Each dataset has its own license and redistribution terms set by its original publisher — see the `README.md` inside each `data/raw/<dataset>/` folder for the exact download source, and [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md#real-datasets) for provenance, DOIs, and checksums (including the CIC-ToN-IoT confirmation release).

## License

This code is released under the [MIT License](LICENSE). Fill in the copyright holder name in `LICENSE` before publishing. Dataset licenses are independent of this code's license and are governed by their original publishers regardless of what license this repository adopts — see [Data availability](#data-availability).

---

## Legacy pipeline (historical)

The sections below describe the **original, superseded** end-to-end pipeline this project started from: a Decision Tree + self-supervised autoencoder hybrid with confidence-based routing, an adaptive threshold, and a temporal error aggregator, evaluated on locally captured traffic plus CICIDS2017/CIC-IDS2018 subsets. It predates the group-aware, real-data-only redesign above and its numbers **must not** be cited — see the evidence-hierarchy note under [Experiment generations](#experiment-generations).

<details>
<summary>Expand for the legacy pipeline's original quick-start, structure, and configuration reference</summary>

### Original novel components

- **Supervised Decision Tree** for known scan attack classification
- **Self-Supervised Autoencoder** for unknown anomaly detection
- **Confidence-Based Routing** — routes flows based on classifier uncertainty
- **Adaptive Confidence Threshold** — adjusts routing dynamically under traffic shift
- **Temporal Error Aggregator** — per-IP rolling error window for early detection
- **SHAP Explainability** — post-hoc interpretability for security operators
- **Calibration Analysis** — ECE-based trustworthy AI evaluation

### Quick start (legacy pipeline)

```bash
pip install -r requirements.txt

# Capture traffic (see capture/ folder), or download public datasets into data/raw/

python main.py                    # full pipeline
python main.py --stage preprocess # or a specific stage
python main.py --stage train
python main.py --stage evaluate

pip install pytest && pytest tests/ -v
```

### Legacy execution order

```
Stage 1:  preprocessing/local/        → local data pipeline
Stage 2:  preprocessing/cicids2017/   → CICIDS2017 alignment
Stage 3:  preprocessing/cic_ids2018/  → CIC-IDS2018 subset + alignment
Stage 4:  models/classifier/          → tune + train classifier
Stage 5:  models/autoencoder/         → train + threshold selection
Stage 6:  models/*/evaluate_*         → individual model evaluation
Stage 7:  baselines/                  → all four baseline evaluations
Stage 8:  hybrid/                     → full hybrid evaluation
Stage 9:  explainability/             → SHAP computation + plots
Stage 10: calibration/                → ECE + confidence distribution
Stage 11: validation/                 → cross-dataset generalization
Stage 12: baselines/compare_baselines → master comparison table
```

### Legacy configuration

All parameters in `config/config.yaml`:
- `hybrid.confidence_threshold` — routing threshold (default: 0.85)
- `adaptive_threshold.window_size` — sliding window size (default: 50)
- `temporal_aggregator.window_size` — per-IP window (default: 10)
- `temporal_aggregator.alert_multiplier` — alert sensitivity (default: 1.5)
- `cic_ids2018.n_scan_samples` — subset size (default: 5000)
- `autoencoder.epochs` — training epochs (default: 50)

### Legacy important rules

1. Never train on CICIDS2017 or CIC-IDS2018 — test only
2. Never refit the scaler on external datasets — `transform()` only
3. Run `tune_classifier.py` before `train_classifier.py`
4. Run `train_autoencoder.py` before `threshold_selector.py`
5. Run all baselines before `compare_baselines.py`

</details>
