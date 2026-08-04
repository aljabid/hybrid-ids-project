# Hybrid Network Intrusion Detection: A Group-Aware Evaluation

**A leakage-controlled, group-aware, multi-domain evaluation of machine-learning intrusion detection — including a hybrid detector whose promising internal and held-out-family results did *not* survive a pre-frozen, untouched external confirmation, on not one but two independent external datasets.**

This repository is the complete, reproducible evidence base for that finding: real network-flow data only, ten paired seeds, an evidence hierarchy that separates split-strategy effects from family exclusion from true cross-domain transfer, and confirmation datasets that were never opened until the method was frozen.

> **The central result is a negative one, on purpose.** A matched random-row-versus-group-aware control shows internal benchmark scores are conditional on split strategy; held-out-family success did not reliably predict cross-domain performance; and the frozen hybrid detector reached MCC ≈ 0 on two untouched, independent confirmation datasets. See [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md) for the full scientific status and the exact claims this evidence does and does not support.

---

## Table of contents

- [What's in this repository](#whats-in-this-repository)
- [The question this project answers](#the-question-this-project-answers)
- [How the method works](#how-the-method-works)
- [The five layers of evidence, with numbers](#the-five-layers-of-evidence-with-numbers)
- [The robustness/ablation/second-confirmation addendum](#the-robustnessablationsecond-confirmation-addendum)
- [Scientific integrity practices used throughout](#scientific-integrity-practices-used-throughout)
- [Independent quality assurance](#independent-quality-assurance)
- [Experiment generations](#experiment-generations)
- [Reproducing the current results](#reproducing-the-current-results)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Data availability](#data-availability)
- [License](#license)
- [Legacy pipeline (historical)](#legacy-pipeline-historical)

---

## What's in this repository

| You want to... | Go to |
|---|---|
| **See the authoritative experimental results** | [`artifacts/redesign_v3/`](artifacts/redesign_v3/) — the current, real-data-only, group-aware experiment generation |
| **Read the manuscript** | `ICECE2026_Manuscript.pdf` (repository root) — submitted for double-blind review to IEEE ICECE 2026 |
| **See the newest evidence (not yet in the manuscript)** | [`artifacts/redesign_v3/ROBUSTNESS_AND_ABLATION_ADDENDUM.md`](artifacts/redesign_v3/ROBUSTNESS_AND_ABLATION_ADDENDUM.md) — a second independent confirmation dataset, bootstrap confidence intervals, and a controlled ablation study |
| **See the pre-submission audit and QA record** | [`artifacts/redesign_v3/PRE_SUBMISSION_AUDIT.md`](artifacts/redesign_v3/PRE_SUBMISSION_AUDIT.md) and [`artifacts/redesign_v3/QA_REPORT.md`](artifacts/redesign_v3/QA_REPORT.md) |
| **Reproduce the core benchmark** | [`benchmark/`](benchmark/) and [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md#reproduction-order) |
| **Understand the project's evolution** | See [Experiment generations](#experiment-generations) below — this project went through three, and only the third is authoritative |
| **Look at the original hybrid-IDS pipeline** | See [Legacy pipeline](#legacy-pipeline-historical) — superseded, kept for history |

---

## The question this project answers

> **Do favorable held-out-family and internal evaluation results reliably predict cross-domain intrusion-detection performance? And where they fail, is the failure a loss of ranking ability, an operating-point/threshold problem, or both?**

The answer, for the datasets, models, and protocol evaluated here, is **no, and it is usually not just a ranking failure** — and the repository is structured to show *why*, not just assert it. Five independently-designed evidence layers converge on the same conclusion from different angles, so that no single failure mode (a bad split, a bad threshold, an unlucky dataset) can be blamed for the whole result.

## How the method works

The system under test is a **Decision Tree–XGBoost agreement rejector**: two independently-tuned branches score every flow, each branch converts its score into a binary call using its own **unlabeled, target-adaptive threshold** (a two-component Gaussian-mixture threshold in logit space for the Decision Tree branch; the target-domain score's median plus six interquartile ranges for the XGBoost branch), and the system predicts an attack only when both branches agree — otherwise it abstains. This is a genuine attempt at a deployable, label-free adaptation mechanism, not a strawman: it never looks at target-domain labels, matching how a real detector would have to operate on a genuinely new network. **No classifier, ensemble, or thresholding rule in this project is claimed as algorithmically novel** — the contribution is the evaluation protocol and the failure analysis, not a new model.

Alongside the hybrid, eight configurations are benchmarked throughout: Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, a family-balanced multiclass Random Forest, a residual tabular network, and a supervised–unsupervised logistic stacker (the "meta hybrid").

## The five layers of evidence, with numbers

Every number below is traceable to a script and a CSV under `artifacts/redesign_v3/` — see that folder's `ARTIFACT_INVENTORY.csv` for exact checksums.

**1. Matched random-row vs. group-aware control.** Holding partition sizes, binary counts, features, hyperparameters, architecture, preprocessing, and all ten seeds fixed, and changing *only* whether CICIDS2017 rows are split by source group or by random row, random-row MCC exceeded group-aware MCC for **all eight** configurations, with the strongest possible non-parametric signal at this sample size (Wilcoxon statistic 0 in every case). The two largest inflations belong to the family-balanced classifier (+0.8215 MCC) and the logistic stacker (+0.6045 MCC) — the two configurations most central to this paper's held-out-family and hybrid claims.

**2. Group-aware CICIDS2017 protocol.** 1,959,228 real flows survived leakage control (55,330 cross-group or conflicting-label hashes removed from 2,111,132 clean candidates), with zero pairwise feature-hash overlap between training, validation, and internal-test partitions, across a shared 68-feature CICFlowMeter-compatible schema.

**3. Held-out-attack-family exclusion.** Every one of the five eligible held-out families (≥5,000 training-partition attack flows; the complete eligible set, not a subsample — DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator) produced positive, moderate-to-high mean MCC for the family-balanced classifier (e.g. DoS Hulk: 0.8530 ± 0.1117), while that same classifier's cross-domain MCC on two independently-collected diagnostic domains remained near zero. Family exclusion within one collection and transfer between collections are different questions with different answers.

**4. Shift-mechanism decomposition.** On CSE-CIC-IDS2018, four of five mandatory models retain ROC-AUC 0.88–0.99 (still strong rankers) while fixed-threshold MCC collapses to at or below zero — an oracle threshold recovers MCC to 0.59–0.98, meaning most of the failure there is a recoverable thresholding problem, and a practical 2,000-flow labeled adaptation budget (Platt scaling) recovers most of that gap in practice. On CIC-UNSW-NB15, ROC-AUC itself is markedly lower (0.57–0.80) and rethresholding recovers far less (0.00–0.31) — a different, more fundamental shift under the identical pipeline. The same pipeline fails for measurably different reasons on two domains both routinely described as "external CICFlowMeter-style traffic."

**5. Frozen external confirmation.** The method was locked (`manifests/final_method_freeze.json`) *before* the confirmation dataset was ever downloaded. On the untouched CIC-ToN-IoT V2 release (4,847,499 real flows), the agreement rejector covered ~69% of flows and reached **MCC −0.0002 ± 0.0006** — indistinguishable from chance. The mechanism is visible in the stored per-seed thresholds: in 9 of 10 seeds the XGBoost branch's unlabeled threshold exceeded 1.0 (outside the valid probability range, predicting "attack" almost never) while the Decision Tree branch's threshold collapsed near zero (predicting "attack" almost always) — the branches agreed almost only on rare mutual "benign" calls. A standard softmax-response selective-classifier baseline confirms this was not an artifact of the rejector's specific engineering: the closest alternative differs by only 0.0058 selective risk at matched coverage.

## The robustness/ablation/second-confirmation addendum

After the manuscript above was finalized, three further evidence-strengthening tasks were completed — none of which change the conclusion, and one of which reinforces it substantially. This work is **not yet integrated into the manuscript PDF**; it lives in [`artifacts/redesign_v3/ROBUSTNESS_AND_ABLATION_ADDENDUM.md`](artifacts/redesign_v3/ROBUSTNESS_AND_ABLATION_ADDENDUM.md) with full manuscript-ready drop-in text, a changelog, and an appendix.

- **A second, fully independent external confirmation on CIC-DDoS2019** (431,371 flows, a dataset from the same UNB/CIC family as CICIDS2017 but a different capture with different attack types, never touched by anything in this project before this test) reproduces the negative result **even more completely**: the deployed hybrid's MCC is **exactly 0.0000 across all 10 of 10 seeds**, and the identical threshold-collapse mechanism recurs (XGBoost branch's threshold invalid in 10 of 10 seeds, vs. 9 of 10 on CIC-ToN-IoT). Every step of the frozen protocol — pooled development-domain training, frozen hyperparameters, the unlabeled threshold rule — was reused unmodified; CIC-DDoS2019 labels were never used for anything but final scoring.
- **Seed-level bootstrap 95% confidence intervals**, extended to the split-strategy, held-out-family, and confirmation results: all eight models' group-aware vs. random-row MCC intervals are non-overlapping, and the deployed confirmation hybrid's MCC interval is a near-point-mass at zero.
- **A four-axis controlled ablation study** (split strategy, held-out-family exclusion, calibration, frozen-vs-oracle confirmation threshold), changing exactly one protocol component at a time. Split strategy is the single largest driver of inflated internal scores, by a wide margin over any other factor tested.

This addendum was independently re-verified in a separate QA pass: every reported number was re-derived from raw CSVs and matched exactly, the full second-confirmation pipeline was re-run from scratch and reproduced byte-identical results, and no data leakage was found in any new code path (see [`QA_REPORT.md`](artifacts/redesign_v3/QA_REPORT.md)).

## Scientific integrity practices used throughout

- **Nothing is tuned on external or confirmation data.** Model selection, hyperparameter search, and thresholding are confined to development data; confirmation datasets are scored, never fit to.
- **The method is frozen before confirmation access**, recorded in a manifest with a timestamp predating the confirmation download.
- **Statistical power limits are stated, not hidden.** With ten paired seeds, the smallest achievable two-sided Wilcoxon p-value is 1/512 ≈ 0.00195; a Holm-corrected family of more than 25 comparisons cannot certify significance at 0.05 regardless of effect size. Where this ceiling binds — including the headline split-strategy result — it is reported explicitly rather than worked around.
- **Negative and null results are reported as found**, not reframed as positive. Where recalibration helps, that is reported; where it hurts (e.g. a sign-flip on CIC-UNSW-NB15), that is reported too.
- **Every manuscript value traces to a packaged CSV, JSON manifest, or checksum** — see `artifacts/redesign_v3/ARTIFACT_INVENTORY.csv`.

## Independent quality assurance

This project underwent two dedicated review passes, both included in the repository for full transparency:

- **[`PRE_SUBMISSION_AUDIT.md`](artifacts/redesign_v3/PRE_SUBMISSION_AUDIT.md)** — an adversarial pre-submission audit (three simulated IEEE reviewer reports, a claim-by-claim evidence-sufficiency attack, and traceability/reproducibility/statistical/formatting/language audits) that treated the manuscript as guilty until proven innocent. It found and fixed a real dataset-label typo in one table and a real mismatch between the documented dependency file and the environment that actually produced the results — both listed with severity ratings alongside everything that held up.
- **[`QA_REPORT.md`](artifacts/redesign_v3/QA_REPORT.md)** — an independent re-verification of the addendum (re-running scripts from scratch and diffing outputs byte-for-byte, tracing every label-access call for leakage) and confirmation that the fixes above work end-to-end from a genuinely clean environment.

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

`requirements-benchmark.txt` pins exact versions, verified against the environment that actually produced every result in this repository — confirmed by installing into a genuinely empty virtual environment and re-running the full test suite (see `artifacts/redesign_v3/QA_REPORT.md` for the verification record; a previous version of this file used version *ranges* that did not match the tested environment, which has been fixed).

Real public datasets are required and are **not** included in this repository (see licensing note below): place CICIDS2017, CSE-CIC-IDS2018, CIC-UNSW-NB15, and the CIC-ToN-IoT V2 confirmation release at the paths documented in [`artifacts/redesign_v3/README.md`](artifacts/redesign_v3/README.md#real-datasets), then follow that file's reproduction order. The matched-split control, shift-mechanism diagnostics, recalibration, and selective-classifier baseline are documented with their exact commands there as well. The robustness/ablation/second-confirmation addendum has its own reproduction commands in [`ROBUSTNESS_AND_ABLATION_ADDENDUM.md`](artifacts/redesign_v3/ROBUSTNESS_AND_ABLATION_ADDENDUM.md), including a Kaggle API token requirement for re-downloading the CIC-DDoS2019 second-confirmation dataset.

## Repository layout

```
hybrid_ids_project/
├── ICECE2026_Manuscript.pdf        # Manuscript, submitted for double-blind review
├── artifacts/redesign_v3/          # Authoritative experiment generation
│   ├── tables/, figures/, manifests/, results/   # The evidence itself
│   ├── README.md                   # Detailed scientific status + reproduction order
│   ├── ROBUSTNESS_AND_ABLATION_ADDENDUM.md   # Second confirmation, bootstrap CIs, ablations
│   ├── PRE_SUBMISSION_AUDIT.md     # Adversarial pre-submission review
│   ├── QA_REPORT.md                # Independent addendum + environment verification
│   └── ARTIFACT_INVENTORY.csv      # Checksummed index of every evidence file
├── benchmark/                      # Generation-2/3 experiment source code (all Python entry points)
├── config/                         # Central configuration (legacy pipeline)
├── data/raw/                       # Where to place downloaded public datasets (not tracked in git)
├── requirements.txt                # Legacy pipeline dependencies
├── requirements-benchmark.txt      # Current benchmark dependencies (exact-pinned)
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

CIC-DDoS2019 (addendum second-confirmation dataset):
Sharafaldin, I., Lashkari, A. H., Hakak, S., & Ghorbani, A. A. (2019). Developing
realistic distributed denial of service (DDoS) attack dataset and taxonomy.
IEEE ICCST 2019.
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
