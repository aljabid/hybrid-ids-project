# Hybrid Network Intrusion Detection

**An honest, leakage-controlled evaluation of machine-learning intrusion detection: how well do models that look excellent in the lab actually work on someone else's network?**

---

## Overview

Machine-learning intrusion detectors routinely report near-perfect scores on public benchmark datasets. Those scores, however, are usually measured on data drawn from the same capture the model was trained on. A detector that recognizes an attack in one traffic collection can miss the same attack in another.

This project builds a rigorous testbed to measure that gap. It trains and evaluates eight detector configurations on real network-flow data, separates three things that are often confused, and reports exactly what each one does and does not prove:

1. **Internal performance:** how well a model does on held-back data from the same collection.
2. **Held-out attack families:** how well it detects an attack type it never saw in training.
3. **True cross-domain transfer:** how well it works on a different dataset, captured on a different network.

It also includes a hybrid detector in which two independent classifiers must agree before a flow is labeled. This detector was frozen before it was tested on external data, so its result is a fair, unbiased test.

## Key findings

- **Evaluation protocol changes the score dramatically.** Splitting data by random row, instead of by source capture group, raised Matthews correlation coefficient (MCC) in all 80 paired comparisons (8 models, 10 seeds). The largest gains were +0.82 for a family-balanced Random Forest and +0.64 for a Decision Tree.
- **Held-out-family success does not predict transfer.** A family-balanced Random Forest detected all five attack families it was never trained on, with mean MCC between 0.47 and 0.85. On two external datasets, the same model's MCC was close to zero.
- **A poor decision does not always mean poor ranking.** On the CIC-DDoS2019 capture, XGBoost ranked flows well (ROC-AUC 0.918). Its frozen threshold produced MCC 0, but the standard 0.5 threshold on the same flows reached MCC 0.335. Comparing decisions on identical flows separates a badly chosen operating point from a real loss of information.
- **The frozen hybrid agreement rule failed on both unseen datasets.** On CIC-ToN-IoT its accepted flows had MCC about 0 (-0.0002), and on CIC-DDoS2019 it labeled every accepted flow benign. The XGBoost branch's unlabeled threshold fell outside the valid probability range in 9 of 10 runs on CIC-ToN-IoT and in all 10 on CIC-DDoS2019, so that branch could never predict an attack.
- **Coverage must be reported.** The agreement rule only makes decisions on a subset of flows (about 69% on CIC-ToN-IoT and 75% on CIC-DDoS2019). Reporting how many flows are covered, and comparing models on the same flows, keeps results honest.

This is a deliberate negative result. No new model is claimed as an improvement. The contribution is the evaluation method and a clear account of how and why detectors fail to transfer.

## Benefits

- **Realistic expectations.** It shows how much of a detector's reported accuracy survives on new traffic, before you rely on it in practice.
- **A reusable, leakage-controlled protocol.** Source-group splits, exact duplicate removal across partitions, and a shared 68-feature schema make comparisons between datasets meaningful.
- **Diagnoses the cause of failure.** Ranking quality (ROC-AUC), decisions at a fixed threshold, calibration, coverage and label-shift diagnostics are reported separately, so you can tell whether a model needs a better threshold, recalibration, or different data.
- **Statistical care.** Ten paired seeds, Wilcoxon signed-rank tests with Holm correction, bootstrap confidence intervals, and explicit statements about what the sample size can and cannot prove.
- **Fully reproducible.** Pinned dependencies, per-seed results, run manifests and an automated test suite (55 tests).

## What was evaluated

**Models (eight configurations):** Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, a family-balanced multiclass Random Forest, a residual tabular neural network, and a logistic stacker that combines supervised and anomaly-detection scores.

**Datasets (real flow data only):**

| Dataset | Role |
|---|---|
| CICIDS2017 | Training, validation and internal testing |
| CSE-CIC-IDS2018 | External development diagnostic |
| CIC-UNSW-NB15 | External development diagnostic |
| CIC-ToN-IoT V2 | Original external confirmation (4.8 million flows) |
| CIC-DDoS2019 | Later external confirmation (431,371 flows) |

**Analyses:** matched random-row versus group-aware splits, leave-one-family-out detection, cross-domain transfer, threshold and prior-shift diagnostics, temperature and Platt recalibration, selective classification with coverage, SHAP feature importance, and computational efficiency.

## Reproducing the results

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-benchmark.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest -q                         # current benchmark: 8 passed
(cd legacy && ../.venv/Scripts/python -m pytest -q)       # original pipeline: 47 passed
```

The datasets are public but are not included in this repository because of their licenses and size. Download them from their original publishers and follow the instructions and manifests under `artifacts/redesign_v3/` to rerun the experiments.

## Limitations

- The collections are synthetic or lab-generated traffic that share feature-extraction conventions. Results may not carry over to a production network.
- Matching column names do not guarantee identical feature meaning across datasets, so some of the observed failure could come from feature differences and not only from attack or network differences.
- With ten seeds, the smallest achievable p-value is too large to pass the Holm correction. Results are descriptive effect sizes, not formal significance claims.
- The held-out-family tasks cover a limited set of attacks, mostly denial-of-service variants.
- Seeds measure training variability only and do not represent independent traffic environments.

## Datasets and citation

If you use this work, please also cite the source datasets:

- **CICIDS2017:** Sharafaldin, Lashkari and Ghorbani (2018). *Toward generating a new intrusion detection dataset and intrusion traffic characterization.* ICISSP.
- **CSE-CIC-IDS2018:** Canadian Institute for Cybersecurity, University of New Brunswick.
- **UNSW-NB15:** Moustafa and Slay (2015). *UNSW-NB15: A comprehensive data set for network intrusion detection systems.* MilCIS.
- **TON_IoT:** Alsaedi, Moustafa, Tari, Mahmood and Anwar (2020). *TON_IoT telemetry dataset.* IEEE Access, 8.
- **CIC-DDoS2019:** Sharafaldin, Lashkari, Hakak and Ghorbani (2019). *Developing realistic distributed denial of service (DDoS) attack dataset and taxonomy.* ICCST.

Dataset licenses are set by their original publishers and are independent of this repository's license.

## License

Released under the [MIT License](LICENSE).
