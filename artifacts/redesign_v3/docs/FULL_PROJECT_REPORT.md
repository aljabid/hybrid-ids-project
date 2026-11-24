# Full Technical Project Report

## Hybrid IDS: Group-Aware, Cross-Domain Robustness, Calibration, and Explainability

This document is an author-facing technical report of the completed project.
It summarizes the implemented system, experimental design, verified results,
limitations, and evidence locations. It is not a manuscript and should not be
submitted as one.

## 1. Project status

The experimental package is complete and has passed its final code and
artifact audit.

- Completion audit: all stages 1–11 complete.
- Automated tests: 55 passed.
- Real data only: yes.
- Mock or synthetic flows: zero.
- Mandatory classifiers: Decision Tree, Random Forest, XGBoost, LightGBM,
  and CatBoost.
- Additional systems: family-aware Random Forest, tabular ResNet, anomaly
  detectors, a calibrated meta-fusion hybrid, and a selective agreement
  hybrid.
- Repeated experiments: ten paired seeds.
- Untouched confirmation: CIC-ToN-IoT V2, 4,847,499 flows.
- Manuscript written or modified by the experimental pipeline: no.
- Package readiness: ready to support manuscript writing.
- Scientific outcome: the frozen selective hybrid failed untouched
  cross-domain confirmation.
- Post-freeze addendum: a matched random-row-versus-group-aware control,
  a focused seed-42 error analysis, a five-model/three-domain reliability
  panel, and exploratory MCC-ECE and JSD-performance comparisons were added
  after this report's initial completion. See §24.

The final machine-readable verdict is in
`manifests/final_readiness_verdict.json`.

## 2. Research objective

The locked objective was to determine whether a semantically aligned,
conservative Hybrid IDS could improve attack detection on held-out
collections and attack families without increasing the false-positive rate by
more than 0.02.

The project intentionally moved beyond random row splitting. It evaluated:

1. Group-aware within-dataset generalization.
2. Cross-dataset transfer.
3. Held-out attack-family detection.
4. Calibration and selective abstention.
5. Explainability and feature drift.
6. Computational cost.
7. One frozen, untouched external confirmation.

The evidence does not support the intended strong superiority hypothesis.
That negative outcome is itself the central scientifically defensible result.

## 3. Data and provenance

### 3.1 CICIDS2017

CICIDS2017 was rebuilt using source CSV files as collection groups. After
cleaning and duplicate/conflict control:

- Clean candidate flows: 2,111,132.
- Final unique flows: 1,959,228.
- Cross-group or conflicting feature hashes removed: 55,330.
- Features: 68.
- Labels: benign plus 14 attack labels.
- Pairwise train/validation/test feature-hash overlap: zero.

Partitions:

| Partition | Rows | Benign | Attack | Attack families | Collection groups |
|---|---:|---:|---:|---:|---:|
| Train | 1,109,433 | 905,134 | 204,299 | 11 | 4 |
| Validation | 587,524 | 459,528 | 127,996 | 1 | 2 |
| Internal group test | 262,271 | 260,423 | 1,848 | 2 | 2 |

The internal test is intentionally difficult and highly imbalanced. It
contains PortScan and Infiltration attacks from source groups that are not
used as ordinary random-row training examples.

### 3.2 CSE-CIC-IDS2018 development diagnostic

- Rows: 593,395.
- Benign: 499,297.
- Attack: 94,098.
- Attack families: 2.
- Features: 68.

This domain was observed during development. It must be described as a
development diagnostic, not as an untouched final test.

### 3.3 CIC-UNSW-NB15 development diagnostic

- Rows: 281,428.
- Benign: 200,258.
- Attack: 81,170.
- Attack families: 9.
- Features: 68.

This domain was also observed during development and is not an untouched
confirmation set.

### 3.4 CIC-ToN-IoT V2 untouched confirmation

- Valid aligned rows: 4,847,499.
- Benign: 2,123,867.
- Attack: 2,723,632.
- Attack families: 9.
- Features: 68.
- Official DOI: `10.48610/f6884ce`.
- Official page:
  `https://staff.itee.uq.edu.au/marius/NIDS_datasets/`.
- Mirror used:
  `https://www.kaggle.com/datasets/dhoogla/cictoniot`.
- Archive SHA-256:
  `c434fca8f461073ebfcf127df5889211ee1b29fba9bf72444091bf0707b4ec2c`.

The cleaned V2 mirror has a different row count from the older summary shown
on the official UQ page. The project records the exact release and checksum;
the manuscript must report the 4,847,499-row release actually evaluated.

## 4. Feature engineering and alignment

The final semantic schema contains 68 CICFlowMeter-compatible features.
Features were included only when names, units, directionality, and semantics
could be aligned across datasets.

Included feature groups:

- Flow duration and packet counts.
- Forward/backward byte and packet-length statistics.
- Flow and directional rates.
- Flow, forward, and backward inter-arrival statistics.
- TCP flag counters.
- Header lengths.
- Packet-length moments.
- Down/up ratio.
- Segment and subflow statistics.
- Active and idle statistics.

Excluded or avoided:

- Timestamp.
- Destination port.
- Protocol.
- Initial-window fields.
- Unstable bulk aggregates.
- Duplicate or semantically ambiguous columns.

The complete mapping is in `tables/semantic_feature_schema.csv`.

### Temporal contribution

No temporal contribution is claimed. Consistent source/session identifiers and
comparable temporal metadata could not be recovered across all domains.

## 5. Preprocessing

Five training-only preprocessing candidates were evaluated under an identical
validation protocol:

1. Raw MinMax scaling.
2. Winsorized StandardScaler with log transforms.
3. Winsorized RobustScaler with log transforms.
4. Winsorized QuantileTransformer with log transforms.
5. Winsorized RobustScaler without log transforms.

The selected pipeline was:

- Winsorization at the 0.1st and 99.9th percentiles.
- `log1p` transformation on 14 heavy-tailed nonnegative features.
- RobustScaler using the 5th–95th percentile range.
- Parameters learned on training data only.

Validation result of the selected transformation:

- MCC: 0.6792.
- TPR: 0.6352.
- FPR: 0.00219.
- PR-AUC: 0.9632.

Evidence: `tables/preprocessing_selection.csv` and
`manifests/preprocessing_selection.json`.

## 6. Models

### 6.1 Mandatory supervised baselines

- Decision Tree.
- Random Forest.
- XGBoost.
- LightGBM.
- CatBoost.

All used the same 68-feature schema, group-aware data protocol, selected
preprocessing, and seed list. Bounded hyperparameter searches used GroupKFold
over CICIDS2017 training collection groups.

Evidence:

- `tables/model_hyperparameters.csv`.
- `tables/group_tuning_summary.csv`.
- `results/group_tuning_*.json`.

### 6.2 Family-aware classifier

A multiclass family-aware Random Forest was trained with family-balanced
sampling. Its binary attack probability was calculated by aggregating
non-benign class probability.

The family-aware objective substantially improved held-out DDoS validation
relative to a binary-balanced Random Forest:

- Family-aware MCC: approximately 0.664.
- Binary-balanced MCC: approximately 0.205.

Evidence: `tables/family_aware_objective_selection.csv`.

### 6.3 Deep tabular baseline

The deep baseline is a residual multilayer network with:

- 68 input features.
- Width 128.
- Three residual blocks.
- Batch normalization.
- Weighted binary cross-entropy.
- AdamW optimization.
- Learning-rate reduction.
- Early stopping and best-checkpoint restoration.

All ten seed histories and checkpoints are present.

### 6.4 Anomaly subsystem

Compared anomaly approaches included:

- Isolation Forest.
- One-class and robust alternatives.
- Denoising autoencoder.
- Robust autoencoder.

The selected development anomaly component was a robust-scaled Isolation
Forest. Validation MCC was approximately 0.179 under the absolute FPR
constraint.

### 6.5 Calibrated meta-fusion hybrid

The first redesigned hybrid used:

- Family-aware classifier.
- Isotonic calibration.
- Isolation Forest anomaly probability.
- Logistic meta-gate.
- Meta-features: classifier probability, anomaly probability, classifier
  entropy, absolute disagreement, and probability product.
- Meta threshold: 0.85.

It improved validation detection but did not remain reliable across domains.

### 6.6 Frozen selective agreement hybrid

The final confirmation candidate combined tuned Decision Tree and XGBoost
branches:

- Decision Tree threshold: a two-component Gaussian mixture in logit score
  space.
- XGBoost threshold: target-score median plus six interquartile ranges.
- Thresholds use the unlabeled target score distribution.
- Output is issued only when branch decisions agree.
- Disagreement produces explicit abstention.

This method was frozen before CIC-ToN-IoT access.

## 7. Experimental protocol

Seeds:

`7, 19, 31, 42, 53, 67, 79, 97, 109, 127`

Primary metrics:

- Accuracy.
- Precision.
- Recall/TPR.
- F1.
- ROC-AUC.
- PR-AUC.
- MCC.
- FPR.
- ECE.
- Brier score.
- Negative log-likelihood.

MCC and PR-AUC should receive more interpretive weight than accuracy because
the internal group test is highly imbalanced.

Statistical procedures:

- Mean ± sample standard deviation over ten paired seeds.
- Paired Wilcoxon signed-rank tests.
- Holm correction for multiple comparisons.
- 20,000-replicate bootstrap confidence intervals.

Selective results condition on covered flows. Coverage and abstained attacks
must always be reported with selective metrics.

## 8. Main benchmark results

### 8.1 CICIDS2017 group test

Ranking by mean MCC:

| Model | MCC | TPR | FPR | PR-AUC |
|---|---:|---:|---:|---:|
| CatBoost | 0.4474 ± 0.0774 | 0.6357 ± 0.1279 | 0.00938 ± 0.00032 | 0.3076 ± 0.0440 |
| XGBoost | 0.4348 ± 0.0513 | 0.4700 ± 0.0693 | 0.00475 ± 0.00024 | 0.3398 ± 0.0422 |
| LightGBM | 0.3666 ± 0.0091 | 0.4474 ± 0.0093 | 0.00710 ± 0.00024 | 0.2434 ± 0.0131 |
| Random Forest | 0.3389 ± 0.0450 | 0.3484 ± 0.0562 | 0.00478 ± 0.00013 | 0.2284 ± 0.0133 |
| Meta hybrid | 0.2882 ± 0.0232 | 0.6868 ± 0.0471 | 0.03328 ± 0.00565 | 0.2350 ± 0.0440 |
| Tabular ResNet | 0.2702 ± 0.1316 | 0.4589 ± 0.2650 | 0.01494 ± 0.00451 | 0.2245 ± 0.0864 |
| Decision Tree | 0.2217 ± 0.0634 | 0.2307 ± 0.0743 | 0.00555 ± 0.00052 | 0.0837 ± 0.0283 |
| Family-aware RF | 0.0954 ± 0.0059 | 0.0844 ± 0.0035 | 0.00441 ± 0.00050 | 0.1056 ± 0.0134 |

Interpretation: CatBoost achieved the best MCC. The meta hybrid achieved the
highest TPR but increased FPR beyond the 0.02 objective and did not produce the
best overall discrimination.

### 8.2 CSE-CIC-IDS2018 development diagnostic

The Decision Tree had the best mean MCC:

- Decision Tree MCC: 0.5111 ± 0.4379.
- Decision Tree TPR: 0.4962 ± 0.4080.
- Decision Tree FPR: 0.0205 ± 0.0239.

The standard deviations are extremely large. This indicates high seed and
threshold instability and must not be summarized as dependable superiority.

The meta hybrid:

- MCC: 0.3853 ± 0.4126.
- TPR: 0.7023 ± 0.4762.
- FPR: 0.2058 ± 0.0585.

Several boosting and forest models had high ranking PR-AUC but near-zero TPR
at a fixed threshold. The author should distinguish ranking performance from
operating-point performance.

### 8.3 CIC-UNSW-NB15 development diagnostic

Best mean MCC:

- LightGBM: 0.1602 ± 0.0943.
- Decision Tree: 0.1593 ± 0.2285.

The meta hybrid:

- MCC: -0.0863 ± 0.1113.
- TPR: 0.00431 ± 0.00117.
- FPR: 0.06365 ± 0.08014.

Interpretation: strong domain-shift failure. The hybrid did not transfer.

## 9. Unknown-family results

Five eligible attack families were held out from training. The family-aware RF
and redesigned meta hybrid were compared over ten paired seeds.

Mean MCC:

| Held-out family ID | Family-aware RF | Meta hybrid | Better |
|---:|---:|---:|---|
| 3 | 0.6578 | 0.5882 | Family-aware RF |
| 4 | 0.8530 | 0.6963 | Family-aware RF |
| 5 | 0.6442 | 0.6125 | Family-aware RF |
| 6 | 0.8273 | 0.8036 | Family-aware RF |
| 7 | 0.4654 | 0.5890 | Meta hybrid |

The hybrid improved mean MCC for only one of five held-out families. It must
not be described as generally superior for unknown attacks.

## 10. Leave-domain-out selective validation

Three-domain leave-one-domain-out evaluation trained on two observed domains
and held out the third.

For the selective agreement hybrid:

- Mean coverage: 0.8652.
- Worst-domain mean MCC: -0.0882.
- Worst-domain mean FPR: 0.0534.

The candidate therefore failed its intended robustness and false-positive
gates before final confirmation.

## 11. Untouched CIC-ToN-IoT confirmation

The frozen method was evaluated once on the complete aligned release for all
ten seeds.

### Full-coverage adaptive Decision Tree

- Coverage: 1.000.
- Accuracy: 0.5497 ± 0.0773.
- Precision: 0.6795 ± 0.1254.
- TPR/Recall: 0.3720 ± 0.1136.
- F1: 0.4750 ± 0.1178.
- ROC-AUC: 0.5751 ± 0.0760.
- PR-AUC: 0.6133 ± 0.0497.
- MCC: 0.1604 ± 0.1633.
- FPR: 0.2225 ± 0.0992.
- ECE: 0.4500 ± 0.0771.
- Brier: 0.4500 ± 0.0771.

### Selective agreement hybrid

- Coverage: 0.6937 ± 0.0772.
- Accuracy on covered flows: 0.4930 ± 0.0537.
- Precision: 0.0499 ± 0.1577.
- TPR on covered flows: 0.000273 ± 0.000863.
- F1: 0.000543 ± 0.001717.
- ROC-AUC: 0.4803 ± 0.0410.
- PR-AUC: 0.5064 ± 0.0690.
- MCC: -0.000199 ± 0.000629.
- FPR on covered flows: 0.000294 ± 0.000930.
- ECE: 0.4474 ± 0.0507.
- Brier: 0.4580 ± 0.0421.

In nine seeds, the selective method issued no attack prediction on covered
flows. In seed 97 it produced 4,929 true positives and 4,958 false positives,
with MCC -0.0020. The low FPR is therefore not evidence of a useful detector;
it primarily reflects collapse toward benign predictions and abstention.

## 12. Calibration

ECE, Brier score, and negative log-likelihood are reported for all models.
Reliability diagrams are supplied for each development domain and for the
confirmation experiment.

The confirmation calibration errors are large. The models should not be
described as trustworthy or well calibrated under domain shift.

Evidence:

- `tables/calibration_results.csv`.
- `tables/confirmation_reliability_bins_seed42.csv`.
- `figures/reliability_*.png`.
- `figures/calibration_comparison.png`.

## 13. Explainability

SHAP artifacts cover eight model variants across four domains:

- Five mandatory tree/boosting models.
- Family-aware Random Forest.
- Tabular ResNet.
- Hybrid supervised backbone.

Total attribution rows: 2,176.

Artifacts:

- `tables/shap_feature_ranking.csv`.
- `tables/shap_top10_features.csv`.
- `tables/cross_dataset_shap_comparison.csv`.
- `figures/global_shap_summary.png`.

The hybrid flow-feature SHAP explanation is explicitly the supervised
classifier backbone. It is not presented as a fabricated end-to-end SHAP
decomposition through the anomaly detector and meta-gate.

## 14. Feature drift

Jensen–Shannon divergence was computed feature-by-feature relative to the
CICIDS2017 training distribution.

Mean JSD:

- CSE-CIC-IDS2018: 0.1089.
- CIC-ToN-IoT confirmation: 0.2048.
- CIC-UNSW-NB15: 0.2145.

Maximum JSD:

- CSE-CIC-IDS2018: 0.3236.
- CIC-ToN-IoT confirmation: 0.6931.
- CIC-UNSW-NB15: 0.6229.

These shifts help explain why apparently strong within-domain scores did not
translate into dependable external operating points.

## 15. Computational efficiency

The isolated efficiency test used 100,000 real CICIDS2017 group-test flows.
Approximate inference latency:

| Model | Microseconds/flow | Peak process RSS (MB) | Model bytes |
|---|---:|---:|---:|
| Decision Tree | 2.08 | 564.1 | 40,858 |
| XGBoost | 4.62 | 583.4 | 263,759 |
| CatBoost | 6.36 | 606.6 | 72,929 |
| LightGBM | 6.72 | 567.3 | 530,909 |
| Random Forest | 6.88 | 567.6 | 2,667,804 |
| Tabular ResNet | 7.69 | 594.7 | 467,259 |
| Family-aware RF | 11.57 | 584.7 | 12,055,147 |
| Meta hybrid | 29.75 | 611.2 | 27,037,334 |

Peak RSS is whole-process peak resident memory, not only incremental model
memory. The meta hybrid is the slowest and largest evaluated system.

## 16. Error analysis

The package contains:

- A confusion matrix for every model/domain seed-42 prediction archive.
- Confirmation confusion matrices for full-coverage and selective variants.
- Family-level false positives and false negatives.
- Top false-positive family table.
- Top false-negative family table.

Evidence:

- `figures/confusion_*.png`.
- `tables/family_error_analysis_seed42.csv`.
- `tables/top_false_positive_families.csv`.
- `tables/top_false_negative_families.csv`.
- `tables/confirmation_family_errors.csv`.

## 17. Statistical robustness

Artifacts include:

- 126 paired baseline comparison rows.
- 54 unknown-family, leave-domain, and confirmation comparison rows.
- Holm-corrected p-values.
- 20,000-replicate bootstrap confidence intervals.

The author must not equate a nonsignificant result with proof of equality.
The paired tests have only ten seed pairs, and selective/full-coverage metrics
have different evaluation scopes.

## 18. Protocol deviation

The v3 protocol contained this stopping rule: if validation gates failed, the
untouched confirmation dataset would not be opened merely to search for a
favorable result.

The validation gates did fail. The project nevertheless proceeded to the
single untouched confirmation because the authorized project objective
explicitly required a final external evaluation. This did not involve
label-driven tuning:

- The method was frozen before download.
- Confirmation labels were not used for revision.
- The confirmation was evaluated once.
- The unfavorable result was retained.

This is a preregistered-protocol deviation, not data leakage. It should be
disclosed in the limitations or experimental-protocol section. Suitable
wording:

> Although the prespecified development gate was not met, we retained the
> already frozen method and performed the planned one-shot external evaluation
> as a confirmatory stress test. No confirmation labels were used for model,
> threshold, or feature revision.

## 19. Claims that are supported

Supported:

- Random row splitting can materially overstate IDS generalization.
- Group-aware evaluation produces lower and more realistic performance.
- Model ranking changes across datasets.
- Family-aware learning can improve some held-out-family conditions.
- The hybrid can increase TPR internally, but often at a substantial FPR cost.
- Calibration and fixed operating thresholds degrade under domain shift.
- Feature drift is substantial across CICFlowMeter-derived datasets.
- Selective agreement can collapse toward abstention/benign prediction.
- External confirmation is essential.
- The frozen candidate failed external confirmation.

## 20. Claims that are prohibited

Do not claim:

- The hybrid consistently outperforms all baselines.
- The hybrid is cross-domain robust.
- The confirmation validated the proposed method.
- The system reliably detects zero-day attacks.
- Low confirmation FPR proves safety.
- The system is well calibrated externally.
- A temporal contribution.
- Real-time deployment readiness based solely on offline timing.
- CSE-CIC-IDS2018 or CIC-UNSW-NB15 was untouched.
- The CIC-ToN-IoT V2 row count equals the older official-page count.
- Statistical significance where Holm-corrected results do not support it.

## 21. Recommended manuscript positioning

The most defensible paper is an evaluation and failure-analysis study, not an
algorithmic-superiority paper.

Recommended focus:

- Leakage-resistant group-aware IDS evaluation.
- Cross-domain operating-point instability.
- Calibration, abstention, and false-positive tradeoffs.
- Unknown-family versus truly external-domain generalization.
- Why a promising validation hybrid failed untouched confirmation.
- Reproducibility and negative-result transparency.

Possible contribution categories:

1. A 68-feature semantic alignment and zero-overlap group protocol.
2. A comprehensive equal-pipeline benchmark over classical, deep, anomaly,
   family-aware, and hybrid systems.
3. Ten-seed group, family, and domain robustness analysis.
4. Calibration, SHAP, drift, efficiency, and family-level error analysis.
5. A frozen one-shot external confirmation demonstrating failure modes of
   selective hybrid IDS.

## 22. Authoritative evidence hierarchy

When two files appear to disagree, use this order:

1. `manifests/final_readiness_verdict.json`.
2. `manifests/claims_audit.json`.
3. `tables/manuscript_results.xlsx`.
4. Final CSV tables in `tables/`.
5. Raw run CSVs in `results/`.
6. Freeze and data manifests.
7. v3 source code in `benchmark/redesign_*.py`.
8. Legacy files only for historical context.

Do not take final numbers from the root legacy `results/`, old notebooks,
`main.py`, `old_main.py`, `artifacts/readiness_v2`, screenshots, or console
logs.

## 23. Package verification

The final package contains:

- Source code.
- Processed real-data arrays.
- Frozen model artifacts.
- Raw run CSVs.
- Final CSV tables.
- A 16-sheet Excel workbook.
- 600-dpi figures.
- Dataset and method manifests.
- Checksummed artifact inventory.
- Reproduction commands.
- Automated tests.

Verification evidence:

- `tables/completion_audit_stages_1_11.csv`.
- `ARTIFACT_INVENTORY.csv`.
- `manifests/final_readiness_verdict.json`.
- `README.md`.

## 24. Matched random-row control (post-freeze addendum)

This section documents work performed after §1-23 were written and after the
package first passed its completion audit. It answers a question the
original package left open: whether split strategy alone changes the
internal score when everything else is held fixed.

### 24.1 Design

For each of the ten seeds, the de-duplicated 1,959,228-flow CICIDS2017 pool
was repartitioned by random row assignment into train, validation, and
internal-test partitions. Each random partition was constructed to have the
exact same row count and the exact same benign/attack count, per partition,
as the corresponding group-aware partition (Table in §3.1). The five
mandatory models (Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost)
were retrained on each random partition using the same 68-feature schema,
the same frozen group-tuned hyperparameters (`results/group_tuning_*.json`),
the same training-only `RobustFlowTransformer` preprocessing specification,
the same 0.5 decision threshold, and the same metric set. Only row
assignment changed. Attack-family mixture was explicitly not forced to
match, because it is a structural consequence of the two split strategies
rather than an independent intervention; this is stated in the manuscript
and in `manifests/matched_split_validation.json`.

An automated manifest check (`benchmark.final_split_comparison.analyze`)
verifies, for every seed, that the random-row partitions have zero pairwise
index overlap and exact row/binary-class-count matches to the group-aware
partitions before writing `manifests/matched_split_validation.json` with
`status: VALIDATED`. The script raises an exception rather than reporting
a result if this check fails.

### 24.2 Result

Random-row MCC exceeded group-aware MCC for every one of the five mandatory
models (mean difference +0.26 to +0.64 MCC; `tables/matched_split_summary.csv`).
All five paired MCC differences, and every TPR/FPR/PR-AUC/ECE difference
reported alongside them (25 comparisons total), were significant after Holm
correction within this comparison family (`tables/matched_split_wilcoxon_holm.csv`,
`p_holm` = 0.048828125, i.e. the manuscript's rounded 0.0488, for every MCC
row). This is the only comparison family in the repository with any
Holm-significant rows; the pre-existing 126 benchmark and 54 robustness
comparisons (§17) have none.

This does not identify a single cause of the gap. Binary counts were
matched by construction, but attack-family mixture and source-group exposure
necessarily differ between random-row and group-aware assignment. The result
supports only the narrower, but still important, claim that internal,
single-domain scores are conditional on the split protocol and should not be
treated as portable estimates of cross-domain behavior. It does not quantify
"how much random-row splitting inflates performance" as a general property of
IDS evaluation, nor does it extend to datasets or model families not tested
here — this remains an open item (see the manuscript's REMAINING_WEAKNESSES.md, maintained outside this repository).

### 24.3 Supplementary analyses

Four additional analyses were produced from the same seed-42 prediction
archives and existing result tables. Each is explicitly scoped as
supplementary or exploratory, per `manifests/final_publication_analysis.json`:

- **Focused error analysis** (`tables/matched_split_seed42_family_errors.csv`):
  family-level false-positive/false-negative/true-positive/true-negative
  counts for all five models under both split strategies, at seed 42 only.
  Illustrative of *why* the aggregate ten-seed MCC gap arises; it does not
  replace the ten-seed statistical result in §24.2.
- **Reliability panel** (`tables/reliability_mandatory_models_seed42.csv`,
  `figures/reliability_mandatory_models.png`): seed-42 confidence-bin
  reliability curves and ECE for all five models across CICIDS2017 group
  test, CSE-CIC-IDS2018, and CIC-UNSW-NB15 (group-aware predictions only).
  Reproduces at the seed level the same internal-versus-external ECE pattern
  already reported at the ten-seed level in §12.
- **MCC-ECE exploratory scatter** (`tables/matched_split_mcc_ece_exploratory.csv`,
  `figures/matched_split_mcc_ece.png`): Spearman rho = -0.4788 (raw
  *p* = 0.1615) across the ten model/split-strategy aggregate points. Not
  Holm-corrected, not significant, and explicitly not interpreted as a
  monotonic calibration-performance relationship — n = 10 aggregate points
  is too low-powered to support that claim.
- **JSD-performance descriptive comparison** (`tables/jsd_performance_exploratory.csv`):
  mean feature-drift JSD alongside median mandatory-model MCC/ECE for
  CSE-CIC-IDS2018 and CIC-UNSW-NB15, plus the CIC-ToN-IoT confirmation
  domain's median ECE. Only two development domains are comparable; a
  correlation estimate is not supported and none is reported.

### 24.4 Reproduction

```bash
.benchmark-venv/bin/python -m benchmark.final_split_comparison
MPLCONFIGDIR=/tmp/mplconfig .benchmark-venv/bin/python \
  -m benchmark.final_publication_analysis
```

`final_split_comparison.run_random_row` resumes from
`results/matched_random_row_ten_seed_runs.csv` if it already contains rows
for a given `(seed, model)` pair, so an interrupted run can be restarted
with the same command.

### 24.5 Manuscript integration

This addendum is integrated into the manuscript as Section IV-A (Table II,
Fig. 2) and Section IV-B (focused error analysis), with the reliability
panel, MCC-ECE scatter, and JSD-performance comparison cited as supplementary
evidence in Section IV-E. It resolves
the manuscript's REMAINING_WEAKNESSES.md item 1 (maintained outside this repository) as originally
numbered ("no matched random-row control"); see that file's "Resolved since
the previous revision" section for the exact scope of what was and was not
resolved.

## 25. Eight-model extension, shift-mechanism decomposition, recalibration, and selective-classifier baseline (second post-freeze addendum)

This section documents work performed after §24, in response to a request to
raise the manuscript from conference-length scope toward a longer-format,
higher-rigor submission. It does not reopen or revise the frozen confirmation
result in §11/§24; it extends the matched-split control and adds three new
diagnostic analyses, all using already-frozen models, already-processed
data, or stored seed-42 predictions.

### 25.1 Eight-model matched-split extension

§24's matched random-row control covered only the five mandatory tree/
boosting models. The family-balanced multiclass Random Forest and the
meta-fusion hybrid — the two configurations the paper's held-out-family and
hybrid claims actually concern — were absent. `benchmark.final_split_comparison_extended`
adds them, plus the tabular ResNet, training each on the same random-row
partitions as §24 with its already-frozen group-aware hyperparameters/
architecture (`manifests/architecture_freeze.json`,
`results/group_tuning_random_forest.json`) reused unmodified.

Result: random-row MCC exceeded group-aware MCC for all eight models, with
Wilcoxon statistic 0 (the maximal possible signal at *n*=10) for MCC in every
case — stronger and more unanimous than the five-model result. However,
expanding the Holm comparison family from 25 tests (5 models x 5 metrics) to
40 tests (8 models x 5 metrics) pushes the required per-test significance
threshold (0.05/40 in the simplest Bonferroni sense, or more precisely the
Holm sequential requirement) below what the achievable *p*-value floor at
*n*=10 (1/512 ≈ 0.001953) can satisfy. The result is `p_holm` = 0.078125
uniformly across all 40 tests — not significant — despite the effect being
unanimous and maximal in every single comparison. This is reported honestly
in the manuscript (Sections III-F, IV-A) as a demonstrated consequence of
correct multiple-comparison correction, explicitly distinguished from the
superseded five-model result (`p_holm` = 0.0488, itself already at the
25-test floor). No comparison family was narrowed after the fact to recover
a "significant" framing.

Evidence: `results/matched_random_row_extended_ten_seed_runs.csv`,
`tables/matched_split_summary.csv` and `tables/matched_split_wilcoxon_holm.csv`
(regenerated for all eight models), `figures/matched_split_mcc.png`
(regenerated), `manifests/matched_split_validation_extended.json`.

### 25.2 Shift-mechanism decomposition

`benchmark.shift_and_calibration_diagnostics.prior_and_label_conditional_drift`
and `.threshold_sensitivity` decompose the marginal JSD already reported in
§14/Table (feature drift) into three components: class-prior shift (odds
ratio), label-conditional covariate shift (JSD computed separately within
the benign and attack subpopulations), and a ranking-versus-operating-point
split (ROC-AUC versus fixed-threshold MCC versus an oracle-best-threshold
MCC computed on target labels — a diagnostic upper bound, not a deployable
rule).

Finding: the two development domains fail for measurably different reasons.
CSE-CIC-IDS2018 has near-zero benign-class drift (mean JSD 0.089) but severe
attack-class drift (mean JSD 0.425); four of five mandatory models retain
ROC-AUC of 0.88-0.99 there, and an oracle threshold recovers MCC from
near-zero to 0.59-0.98 — the collapse is overwhelmingly a fixed-threshold
problem. CIC-UNSW-NB15 has more balanced, moderate drift across both classes
(benign JSD 0.244, attack JSD 0.315); ROC-AUC there is markedly lower
(0.57-0.80) and oracle rethresholding recovers far less (0.00-0.31 MCC) —
a genuine ranking-ability loss, not merely a threshold problem. The
confirmation domain's own class-prior shift (odds ratio 5.70, the largest of
any evaluated domain) is also newly quantified here, though it was not used
to revise the frozen confirmation configuration.

Evidence: `tables/prior_shift_summary.csv`, `tables/label_conditional_feature_drift.csv`,
`tables/threshold_sensitivity_seed42.csv`, `figures/label_conditional_drift.png`,
`manifests/shift_and_calibration_diagnostics.json`.

### 25.3 Recalibration

`benchmark.shift_and_calibration_diagnostics.recalibration_experiment` fits
temperature scaling [Guo et al., already cited] and Platt scaling on a
2,000-flow stratified adaptation fold sampled from stored seed-42
predictions, evaluated on the disjoint remainder (279,428-591,395 flows
depending on domain). Temperature scaling, being a monotonic one-parameter
transform, leaves 0.5-threshold MCC/TPR/FPR unchanged by mathematical
construction (verified bit-for-bit equal in the output CSV for all ten
model/domain rows) and can only change ECE/Brier/NLL; its ECE effect is
domain-dependent (helps CIC-UNSW-NB15 for four of five models, worsens
CSE-CIC-IDS2018 for all five). Platt scaling, a two-parameter transform that
can shift the decision boundary, recovers most of §25.2's oracle-threshold
gap in practice for CSE-CIC-IDS2018 (e.g. Random Forest MCC -0.0018 to
0.8725; LightGBM -0.0287 to 0.9741) but is unreliable on CIC-UNSW-NB15,
reducing MCC for two of five models even as it reduces their ECE. This
result is reported for both directions (helpful and harmful) rather than
only its successes, and is explicitly scoped as a seed-42 illustration with
one arbitrary adaptation-fold draw, not a validated ten-seed procedure.

Evidence: `tables/recalibration_seed42.csv`, `figures/recalibration_ece_comparison.png`.

### 25.4 Selective-classifier baseline

`benchmark.selective_classifier_baseline` builds a standard softmax-response
(Chow's rule) risk-coverage curve on the frozen confirmation-stage DT and
XGBoost branch probabilities already used for §11's agreement rejector,
using three confidence signals (DT-only distance from 0.5, XGBoost-only
distance from 0.5, and a continuous "graded disagreement" generalization of
the engineering rule's binary agreement test). The engineering rejector's
realized (coverage, risk) point is plotted against these curves. The
baselines deliberately use the standard 0.5 decision boundary rather than
the frozen rejector's own target-adaptive thresholds, which are themselves
extreme (see the mechanism note below); this asks whether a conventionally
centered rule does better, not whether the frozen thresholds could be
salvaged with a softer accept/reject boundary.

**Mechanism of the nine-of-ten all-benign collapse.** Inspecting the stored
per-seed threshold values in `results/confirmation_ten_seed_runs.csv`
(post-hoc observation, not a designed experiment) explains a pattern already
reported in §11's README bullet ("in nine of ten confirmation seeds, the
selective system predicted every covered flow as benign"): in nine of ten
seeds, the XGBoost branch's target-adaptive threshold (median plus six
interquartile ranges of unlabeled target scores) exceeds 1.0 — outside the
valid probability range — so that branch predicts "attack" for essentially
no flow. Simultaneously, the Decision Tree branch's Gaussian-mixture
threshold collapses to approximately zero, so that branch predicts "attack"
for essentially every flow. The two branches agree almost exclusively where
XGBoost's near-universal "benign" call happens to coincide with a rare
Decision Tree "benign" call — hence the near-total collapse to benign
predictions. Seed 97, the only seed with a nonzero confirmation MCC, is also
the only seed where the XGBoost threshold remained inside the valid
probability range (0.967).

Finding: none of the three principled baselines meaningfully outperforms the
engineering rejector at matched coverage (largest gap 0.0364 selective risk,
DT-only, below its realized 0.5522; smallest gap 0.0058, XGBoost-only); all
remain close to or worse than a trivial always-predict-majority-class
classifier's 0.436 error rate given the domain's 0.564 attack prior. This
shows the confirmation-domain failure
reported in §11 is not an artifact of the specific engineering rejector's
design — a differently designed but equally standard selective rule
collapses to comparable risk. The curves also show the DT branch alone
carries some real signal at very low coverage (risk falls to roughly
0.11-0.15, non-monotonically, across the 4-12% coverage band) that the
two-branch agreement design does not exploit,
though acting on this would require redesigning and refreezing the
confirmation configuration, which this diagnostic does not do.

Evidence: `tables/selective_classifier_risk_coverage_confirmation.csv`,
`tables/selective_classifier_summary_confirmation.csv`,
`figures/selective_risk_coverage_confirmation.png`,
`manifests/selective_classifier_baseline.json`.

### 25.5 Held-out-family scope and second-confirmation-domain feasibility

Two open items from the manuscript's REMAINING_WEAKNESSES.md (maintained
outside this repository) were investigated rather than left as unexamined
assertions:

- **Held-out-family scope**: verified that exactly five CICIDS2017
  training-partition attack families meet the repository's own >=5,000-flow
  eligibility bar (the sixth-largest family has 3,020 flows). This is a hard
  data ceiling under the existing eligibility rule, not an unexploited
  opportunity to test more families against the current dataset.
- **Second untouched confirmation domain**: no Kaggle API credentials are
  configured on the development machine (`kaggle.json` absent), and both the
  official UQ NIDS-dataset repository (`staff.itee.uq.edu.au`) and UNB's
  direct CIC dataset download infrastructure require an authenticated
  browser session that a script cannot complete (confirmed by direct HTTP
  probing during this work). This blocks acquiring a second untouched
  confirmation domain on infrastructure/credentials grounds, not analysis
  effort.

### 25.6 Reproduction

```bash
python -m benchmark.final_split_comparison_extended
python -m benchmark.shift_and_calibration_diagnostics
python -m benchmark.selective_classifier_baseline
```

(See README.md's "Environment note" for which Python interpreter is
actually functional on a given machine.) The extended matched-split training
run took approximately 90 minutes on the reference CPU machine, dominated by
tabular ResNet training (up to ~90 epochs per seed with early stopping); the
other two scripts complete in under a minute each without CPU contention.

### 25.7 Manuscript integration

Integrated as: Table II and Fig. 2 (eight-model matched split, Section
IV-A); Table V and Fig. 4 (shift mechanism, Section IV-F); Table VI and
Fig. 5 (recalibration, Section IV-G); Table VIII and Fig. 6 (selective
baseline, Section IV-I). The frozen confirmation table (previously Table V)
is renumbered to Table VII as a consequence of the new sections' insertion
ahead of it in reading order. This resolves or substantially addresses
`REMAINING_WEAKNESSES.md` items on shift mechanism, calibration, and
selective classification (see that file's "Resolved or substantially
addressed" section for the precise scope of each), and documents concrete
findings — rather than only restating the open question — for held-out-family
scope and second-confirmation-domain feasibility.

