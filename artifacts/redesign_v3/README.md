# Hybrid IDS Redesign v3 — Reproducibility Package

This directory contains the real-data-only experimental package produced for
the group-aware, cross-domain Hybrid IDS study. It contains numerical results,
processed arrays, frozen models, manifests, CSV/Excel tables, and 600-dpi
figures. It does not contain manuscript text.

A post-manuscript addendum — a second, independent external confirmation
on CIC-DDoS2019 through the identical frozen protocol, bootstrap confidence
intervals, calibration-sensitivity sweeps, and a four-axis controlled
ablation study, none of which retrains or modifies the frozen method — is
documented separately in
[`ROBUSTNESS_AND_ABLATION_ADDENDUM.md`](ROBUSTNESS_AND_ABLATION_ADDENDUM.md).
The second confirmation reproduces the paper's central negative result
(deployed hybrid MCC = 0.0000 exactly across all 10 seeds on CIC-DDoS2019)
and reproduces its mechanistic cause (the XGBoost branch's unlabeled
threshold heuristic again lands outside the valid probability range, in
10 of 10 seeds).

## Scientific status

The evaluation is complete, but it does **not** support a claim that the
hybrid is consistently superior or cross-domain robust.

- Development evaluation used ten seeds and three observed domains.
- Unknown-family evaluation held out five eligible CICIDS2017 attack families.
- The final method was frozen before CIC-ToN-IoT was downloaded.
- The untouched CIC-ToN-IoT confirmation used 4,847,499 valid real flows.
- Mean selective coverage was 0.6937.
- Selective confirmation MCC was -0.0002 ± 0.0006.
- In nine of ten confirmation seeds, the selective system predicted every
  covered flow as benign. In seed 97 it detected 4,929 attacks but also
  produced 4,958 false positives, with MCC -0.0020.
- The full-coverage adaptive Decision Tree achieved confirmation MCC
  0.1604 ± 0.1633, TPR 0.3720 ± 0.1136, and FPR 0.2225 ± 0.0992.

These limitations are first-class results and must be reported. The project
must not claim universal superiority, robust zero-day detection, or successful
external confirmation.

## Environment note

The reproduction commands below (and elsewhere in this file) reference
`.benchmark-venv/bin/python`, a Linux-style venv created for this project. On
a Windows checkout that virtual environment's `bin/python` symlink can be
broken (it targets `/usr/bin/python`, which is not necessarily resolvable
outside the environment it was created in). The native Windows
`.venv\Scripts\python.exe` at the repository root already has the full
dependency set installed and is the environment actually used to run every
script in this package, including the eight-model matched-split extension,
the shift-mechanism/recalibration/selective-classifier scripts below, and
the robustness/ablation/second-confirmation addendum. `python -m pytest
tests/ -q` passes 55/55 under it. As of the QA pass recorded in
`QA_REPORT.md`, `requirements-benchmark.txt` (repository root) pins exact
versions matched against this `.venv` — that is now the canonical,
verified reproduction path; `.benchmark-venv` is left below for historical
reference only and is not guaranteed to resolve on every platform. Use
whichever environment is actually functional on your machine; the module
invocations (`python -m benchmark.X`) are identical either way.

## Matched random-row control (added after the initial v3 freeze; extended to eight models below)

A follow-up experiment answers the question the original package left open:
does split strategy alone, holding everything else fixed, change the
internal score? For every seed and every mandatory model, the de-duplicated
CICIDS2017 pool was repartitioned by random row assignment into train,
validation, and internal-test partitions with the exact same row counts and
benign/attack counts as the existing group-aware partitions. Features,
frozen group-tuned hyperparameters, preprocessing specification, decision
threshold, metrics, and the ten-seed list were unchanged; only row assignment
differed.

- Random-row MCC exceeded group-aware MCC for all five mandatory models.
- Every model's MCC difference remained significant after Holm correction
  across 25 paired comparisons (`p_holm` = 0.0488 for every MCC row).
- The gap does not identify a single cause: binary counts were matched, but
  attack-family mixture and source-group exposure differ by construction
  between the two split strategies.
- This does not retroactively validate any development-domain or
  confirmation-domain claim above; it only shows that internal, single-domain
  scores are conditional on the split protocol.

**This control was subsequently extended to all eight benchmarked
configurations** (family-aware RF, meta hybrid, tabular ResNet added to the
five mandatory models above) — see "Eight-model extension" below. The
25-comparison Holm result above is superseded by the 40-comparison result in
that section; both are retained here because the manuscript explicitly
discusses the difference between them as a demonstration of how family size
affects achievable significance (see the manuscript's editorial changelog,
maintained outside this repository, Revision 3 entry 32).

A focused seed-42 error case study, a five-model/three-domain reliability
panel, an exploratory MCC-versus-ECE Spearman analysis, and a descriptive
JSD-versus-performance comparison were produced alongside it. All four are
supplementary/exploratory: none introduces a new claim beyond what is stated
in `manifests/final_publication_analysis.json`, and the JSD-performance and
MCC-ECE analyses explicitly report low statistical power (n = 2-3 and n = 10
respectively) rather than a validated relationship.

Evidence:

- `results/matched_random_row_ten_seed_runs.csv`: 50 seed-level rows (5
  models x 10 seeds).
- `tables/matched_split_summary.csv`: mean/SD per model and split strategy.
- `tables/matched_split_wilcoxon_holm.csv`: paired Wilcoxon/Holm results.
- `tables/matched_split_partition_validation.csv`: proof of matched row and
  binary counts with zero cross-partition overlap, per seed.
- `tables/matched_split_seed42_family_errors.csv`: five-model, two-split
  family-level false-positive/false-negative counts.
- `tables/reliability_mandatory_models_seed42.csv` and
  `figures/reliability_mandatory_models.png`: seed-42 reliability panel.
- `tables/matched_split_mcc_ece_exploratory.csv` and
  `figures/matched_split_mcc_ece.png`: exploratory calibration-discrimination
  scatter.
- `tables/jsd_performance_exploratory.csv`: descriptive JSD-versus-MCC/ECE
  comparison.
- `manifests/matched_split_validation.json`: automated pass/fail check on the
  matching contract (raises an error rather than reporting a result if the
  contract were violated).
- `manifests/final_publication_analysis.json`: completion record and stated
  scope limitations for the four supplementary analyses.
- Reproduce with `python -m benchmark.final_split_comparison` followed by
  `python -m benchmark.final_publication_analysis` from the repository root
  (see "Environment note" above for which Python to use).

## Eight-model extension of the matched random-row control

The five-mandatory-model control above omitted the two configurations most
central to this project's held-out-family and hybrid claims: the
family-balanced multiclass Random Forest and the meta-fusion hybrid. This
extension adds them, plus the tabular ResNet, reusing each configuration's
already-frozen group-aware hyperparameters/architecture (from
`manifests/architecture_freeze.json` and
`results/group_tuning_random_forest.json`) so that split strategy remains the
only varied factor.

- Random-row MCC exceeded group-aware MCC for all eight configurations.
- For MCC specifically, the Wilcoxon statistic was 0 (the strongest possible
  one-directional signal at ten paired seeds) in all eight cases — a
  *stronger* and more unanimous signal than the five-model result.
- **This does not remain Holm-significant.** The comparison family is now 8
  models x 5 metrics = 40 tests. At ten paired seeds the smallest achievable
  two-sided Wilcoxon *p*-value is 1/512 ≈ 0.001953; a 40-test Holm correction
  requires *p* < 0.00125 per test to certify significance at 0.05, which the
  achievable floor cannot reach. The result: `p_holm` = 0.078125 uniformly,
  not significant, despite the effect being unanimous and maximal in
  magnitude. This is reported as a real consequence of correct multiple-
  comparison correction, not walked back to the smaller five-model family to
  preserve a "significant" headline.

Evidence:

- `results/matched_random_row_extended_ten_seed_runs.csv`: 30 seed-level rows
  (3 extended models x 10 seeds).
- `tables/matched_split_summary.csv` / `tables/matched_split_wilcoxon_holm.csv`:
  regenerated to cover all eight models (overwrites the five-model-only
  version; the five-model raw numbers are unchanged, only the aggregation
  and correction family grew).
- `figures/matched_split_mcc.png` (+`.pdf`/`.svg`): regenerated with all eight
  models, group-aware/random-row bars, a dotted line separating the five
  mandatory models from the three extensions.
- `manifests/matched_split_validation_extended.json`: validation manifest for
  the eight-model extension, including the Holm-correction explanation.
- Reproduce with `python -m benchmark.final_split_comparison_extended` (runs
  `run_extended()` then `analyze_all()`; resumable per (seed, model) via the
  same checkpoint pattern as the five-model script). Training all eight
  models across ten seeds took approximately 90 minutes on the reference CPU
  machine, dominated by the tabular ResNet (up to ~90 epochs per seed with
  early stopping).

## Shift-mechanism, recalibration, and selective-classifier baseline

Three further analyses, added in the same revision as the eight-model
extension, decompose *why* cross-domain scores collapse and whether cheap
interventions repair it. None retrains any model; all reuse stored seed-42
predictions and existing processed feature arrays.

**Shift mechanism** (`benchmark/shift_and_calibration_diagnostics.py`):

- Class-prior odds ratio relative to CICIDS2017 training: 0.82
  (CSE-CIC-IDS2018), 1.80 (CIC-UNSW-NB15), 5.70 (CIC-ToN-IoT confirmation).
- Label-conditional JSD (benign-only vs. attack-only) shows CSE-CIC-IDS2018's
  drift is strongly attack-class-concentrated (benign JSD 0.089, attack JSD
  0.425), while CIC-UNSW-NB15 and the confirmation domain are more balanced
  across classes.
- An oracle-best-threshold search (using target labels; a diagnostic upper
  bound, not a deployable rule) shows CSE-CIC-IDS2018's MCC collapse is
  overwhelmingly a rethresholding-recoverable problem (ROC-AUC 0.88-0.99
  preserved; oracle MCC recovers to 0.59-0.98 for four of five models), while
  CIC-UNSW-NB15's collapse includes genuine ranking loss (ROC-AUC 0.57-0.80;
  oracle recovery only 0.00-0.31 MCC). Two development domains, two
  different shift mechanisms, under one identical pipeline.
- Evidence: `tables/prior_shift_summary.csv`, `tables/label_conditional_feature_drift.csv`,
  `tables/threshold_sensitivity_seed42.csv`, `figures/label_conditional_drift.png`,
  `manifests/shift_and_calibration_diagnostics.json`.

**Recalibration** (same script): temperature scaling and Platt scaling fit on
a 2,000-flow stratified adaptation fold, evaluated on the disjoint remainder.
Temperature scaling is monotonic and leaves MCC unchanged by mathematical
construction (verified bit-for-bit in the output CSV); its ECE effect is
mixed (helps CIC-UNSW-NB15, hurts CSE-CIC-IDS2018 for all five models). Platt
scaling recovers most of the CSE-CIC-IDS2018 oracle-threshold gap in
practice (e.g. Random Forest MCC -0.0018 to 0.8725) but actively reduces MCC
for two of five models on CIC-UNSW-NB15 — reported as-is, not selectively.
Evidence: `tables/recalibration_seed42.csv`, `figures/recalibration_ece_comparison.png`.

**Selective-classifier baseline** (`benchmark/selective_classifier_baseline.py`):
a standard softmax-response (Chow's rule) risk-coverage baseline on the
frozen confirmation-stage DT/XGBoost branch probabilities, compared against
the engineering agreement rejector's realized (coverage, risk) point. None of
three principled variants (DT-only, XGBoost-only, graded disagreement) beats
the engineering rejector by more than 0.036 selective risk at matched
coverage (DT-only: 0.0364 lower; graded disagreement: 0.0289 lower; XGBoost-only:
0.0058 lower) — the confirmation-domain failure is not an artifact of that
specific rejector's design. Evidence:
`tables/selective_classifier_risk_coverage_confirmation.csv`,
`tables/selective_classifier_summary_confirmation.csv`,
`figures/selective_risk_coverage_confirmation.png`,
`manifests/selective_classifier_baseline.json`.

Building this baseline surfaced a mechanism for a previously-unexplained
pattern (the "nine of ten seeds predicted every covered flow as benign"
bullet above): `results/confirmation_ten_seed_runs.csv` shows the XGBoost
branch's target-adaptive threshold (median + 6*IQR of unlabeled target
scores) exceeds 1.0 — outside the valid probability range — in nine of ten
seeds, so that branch predicts "attack" almost never; the Decision Tree
branch's threshold simultaneously collapses to ~0, predicting "attack"
almost always. Agreement between two branches leaning to opposite extremes
happens mostly on the rare cases both call "benign." Seed 97, the only seed
with any covered detections, is also the only seed where the XGBoost
threshold stayed inside [0,1] (0.967). See `FULL_PROJECT_REPORT.md` §25.4
for the full account; this is a post-hoc mechanistic observation from
already-stored data, not a new experiment.

Reproduce with, in order: `python -m benchmark.shift_and_calibration_diagnostics`
then `python -m benchmark.selective_classifier_baseline`. Both complete in
well under a minute once run without CPU contention from the extended
training job above; expect several minutes if run concurrently with it.

## Key locations

- `results/matched_random_row_ten_seed_runs.csv` /
  `results/matched_random_row_extended_ten_seed_runs.csv`: matched random-row
  control, all 8 models × 10 seeds combined.
- `tables/matched_split_summary.csv` /
  `tables/matched_split_wilcoxon_holm.csv`: matched-split aggregates and
  paired significance tests, all 8 models.
- `tables/threshold_sensitivity_seed42.csv` / `tables/recalibration_seed42.csv`:
  shift-mechanism and recalibration results.
- `tables/selective_classifier_summary_confirmation.csv`: selective-classifier
  baseline vs. engineering rejector.
- `results/grouped_ten_seed_runs.csv`: 8 models × 3 domains × 10 seeds.
- `results/unknown_family_ten_seed_runs.csv`: paired unknown-family tests.
- `results/leave_domain_out_selective_runs.csv`: leave-domain-out validation.
- `results/confirmation_ten_seed_runs.csv`: frozen one-shot confirmation.
- `tables/manuscript_results.xlsx`: consolidated 16-sheet workbook.
- `tables/benchmark_mean_std.csv`: benchmark mean and standard deviation.
- `tables/model_hyperparameters.csv`: fixed search results.
- `tables/calibration_results.csv`: ECE, Brier, and log loss.
- `tables/computational_efficiency.csv`: latency, peak RSS, and model size.
- `tables/statistical_tests_wilcoxon_holm.csv`: baseline paired tests.
- `tables/robustness_statistical_tests_wilcoxon_holm.csv`: unknown-family,
  leave-domain, and confirmation tests.
- `tables/bootstrap_confidence_intervals.csv`: 20,000-replicate intervals.
- `tables/shap_top10_features.csv`: Top-10 SHAP features for every model.
- `tables/cross_dataset_shap_comparison.csv`: cross-domain SHAP comparison.
- `tables/feature_drift_jsd.csv`: quantitative feature drift.
- `manifests/final_method_freeze.json`: pre-confirmation method freeze.
- `manifests/confirmation_execution.json`: proof of 10-seed completion.
- `manifests/claims_audit.json`: permitted and prohibited claims.
- `ARTIFACT_INVENTORY.csv`: artifact checksums and sizes.

## Environment

From the repository root:

```bash
python -m venv --system-site-packages .benchmark-venv
.benchmark-venv/bin/pip install -r requirements-benchmark.txt
```

The recorded run used Python 3.13 and CPU execution. PyArrow is required for
streaming the CIC-ToN-IoT Parquet file.

## Real datasets

Place the datasets at:

```text
data/raw/cicids2017/
data/raw/cic_ids2018/
data/raw/cic_unsw_nb15/
data/raw/cic_ton_iot_confirmation/CIC-ToN-IoT-V2.parquet
```

CIC-ToN-IoT provenance:

- Official DOI: `10.48610/f6884ce`
- Official UQ page: `https://staff.itee.uq.edu.au/marius/NIDS_datasets/`
- Public mirror used because the current UQ file link returned its application
  shell: `https://www.kaggle.com/datasets/dhoogla/cictoniot`
- Download archive SHA-256:
  `c434fca8f461073ebfcf127df5889211ee1b29fba9bf72444091bf0707b4ec2c`

The mirror's cleaned V2 contains 4,847,499 rows, which differs from the older
count displayed on the official UQ page. The exact source, count, and checksum
are recorded rather than silently reconciling the releases.

## Reproduction order

Run from the repository root:

```bash
.benchmark-venv/bin/python -m benchmark.redesign_data
.benchmark-venv/bin/python -m benchmark.redesign_preprocessing
.benchmark-venv/bin/python -m benchmark.redesign_supervised
.benchmark-venv/bin/python -m benchmark.redesign_hybrid
.benchmark-venv/bin/python -m benchmark.redesign_validation
.benchmark-venv/bin/python -m benchmark.redesign_unknown_family
.benchmark-venv/bin/python -m benchmark.redesign_selective
```

Only after `manifests/final_method_freeze.json` exists, build and run the
confirmation:

```bash
.benchmark-venv/bin/python -m benchmark.redesign_confirmation
```

On memory-limited systems, the same run can be resumed by seed and phase:

```bash
.benchmark-venv/bin/python -m benchmark.redesign_confirmation \
  --seed 127 --phase sample-domain --domain cicids2017
.benchmark-venv/bin/python -m benchmark.redesign_confirmation \
  --seed 127 --phase transform
.benchmark-venv/bin/python -m benchmark.redesign_confirmation \
  --seed 127 --phase prepare-dt
.benchmark-venv/bin/python -m benchmark.redesign_confirmation \
  --seed 127 --phase prepare-xgb
.benchmark-venv/bin/python -m benchmark.redesign_confirmation \
  --seed 127 --phase score-dt
.benchmark-venv/bin/python -m benchmark.redesign_confirmation \
  --seed 127 --phase score-xgb
.benchmark-venv/bin/python -m benchmark.redesign_confirmation \
  --seed 127 --phase finalize
.benchmark-venv/bin/python -m benchmark.redesign_confirmation --phase finish
```

The scoring phases are row-checkpointed. Repeat a scoring command until it
reports `scored`; incomplete work resumes from its stored row offset.

Generate final analyses:

```bash
for model in decision_tree random_forest xgboost lightgbm catboost \
  family_aware_rf redesigned_meta_hybrid tabular_resnet; do
  MPLCONFIGDIR=/tmp/mplconfig .benchmark-venv/bin/python \
    -m benchmark.redesign_explainability --model "$model"
done
MPLCONFIGDIR=/tmp/mplconfig .benchmark-venv/bin/python \
  -m benchmark.redesign_explainability --model finalize

for model in decision_tree random_forest xgboost lightgbm catboost \
  family_aware_rf redesigned_meta_hybrid tabular_resnet; do
  .benchmark-venv/bin/python -m benchmark.redesign_efficiency --model "$model"
done
.benchmark-venv/bin/python -m benchmark.redesign_efficiency --model finalize
MPLCONFIGDIR=/tmp/mplconfig .benchmark-venv/bin/python \
  -m benchmark.redesign_analysis
```

Generate the matched random-row control and its dependent supplementary
analyses (requires `results/group_tuning_*.json` and
`manifests/final_method_freeze.json` from the steps above):

```bash
.benchmark-venv/bin/python -m benchmark.final_split_comparison
MPLCONFIGDIR=/tmp/mplconfig .benchmark-venv/bin/python \
  -m benchmark.final_publication_analysis
```

`final_split_comparison.run_random_row` is resumable: it skips any
`(seed, model)` pair already present in
`results/matched_random_row_ten_seed_runs.csv`, so an interrupted run can be
restarted with the same command. `final_publication_analysis` requires the
seed-42 prediction archives written by both the group-aware benchmark and
`final_split_comparison` (`results/predictions/*_seed_42.npz`).

## Interpretation safeguards

- External diagnostic domains were observed during development and are not
  described as untouched tests.
- CIC-ToN-IoT is the only untouched one-shot confirmation dataset.
- Selective metrics condition on covered flows; coverage and abstained attacks
  must always accompany them.
- The matched random-row control isolates split strategy under matched row
  and binary-class counts; it does not isolate attack-family composition,
  which differs by construction between the two split strategies, and it
  covers only CICIDS2017 and the five mandatory models.
- The MCC-ECE and JSD-performance comparisons are exploratory and
  low-n (n = 10 and n = 2-3 respectively); neither supports a causal or
  significant-correlation claim.
- The 16,384-bin confirmation ROC/PR implementation processes every row.
  Validation against exact sorting differed by approximately `1.4e-7` for
  ROC-AUC and `7.4e-6` for PR-AUC on 100,000 values.
- SHAP uses only real, reproducibly seeded samples. The hybrid flow-feature
  explanation is explicitly its supervised backbone, not a fabricated
  end-to-end attribution through the anomaly and meta-gate stages.
- No temporal contribution is claimed because consistent flow-time/session
  metadata could not be recovered across domains.
