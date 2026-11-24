# Robustness and Ablation Addendum (post-manuscript)

Status: Tasks 1, 2, and 3 all complete.

Scope and integrity constraints honored throughout this addendum:

- The frozen method (`manifests/final_method_freeze.json`) was not touched.
  No model was retrained. No hyperparameter, threshold, or feature was
  changed.
- Nothing here is tuned on an external/confirmation dataset. The one new
  computation on confirmation data (Axis 4, below) is an explicitly-labeled,
  non-deployable diagnostic that does not alter the actual frozen decision
  rule or any previously reported confirmation number.
- No result is cherry-picked or hidden. Section "Findings that complicate
  the story" below exists specifically to surface the two results that did
  not simply confirm what was already believed.
- Every number in this document is traceable to a script and a CSV; see
  [Traceability](#traceability).

## Task 1: A second, independent external dataset

### Dataset selection and access

Three candidate access paths were investigated for a genuinely independent,
CICFlowMeter-schema-compatible dataset. UNB's historical direct-download
file server (`205.174.165.80` / `cicresearch.ca`) now returns the same
site-template HTML for every path tested — the raw file host has been
decommissioned. The Kaggle API requires credentials the environment did not
initially have. Two candidates from the task's own suggested list were
screened out before attempting download: TON-IoT/NF-ToN-IoT is the same
underlying capture as the already-frozen CIC-ToN-IoT confirmation dataset
(leakage risk), and UNSW-NB15 is already an in-repo development-diagnostic
domain (not untouched).

The user then supplied a Kaggle API token (new-style token, saved to
`~/.kaggle/access_token`, authenticated via the official `kaggle` package —
not the legacy `username`+`key` pair). **CIC-DDoS2019** (Kaggle mirror
`dhoogla/cicddos2019`, 431,371 rows across 17 per-attack-type files) was
selected: it is public, well-cited, flow-based, and — critically — hosted by
the same curator ("dhoogla") whose CIC-ToN-IoT mirror is already the
project's frozen confirmation dataset, so it uses the identical modernized
CICFlowMeter column-naming convention. Confirmed directly: the existing,
unmodified `benchmark.redesign_confirmation.column_mapping` function (built
for CIC-ToN-IoT) maps all 68 canonical features from CIC-DDoS2019's raw
columns with **zero new aliases needed** — strong, checked evidence of
genuine feature-semantic compatibility rather than an assumption.

### Protocol (identical, not modified)

`benchmark/second_confirmation_cicddos2019.py` reuses, unmodified and by
import, every component of the frozen method: the 68-feature canonical
schema, the column mapping, `domain_balanced_sample` (pooled sampling from
CICIDS2017 / CSE-CIC-IDS2018 / CIC-UNSW-NB15 — the same three development
domains used everywhere else), `tuned_pipeline` (the same frozen
hyperparameters from `group_tuning_*.json`), and `predict_selective` (the
same unlabeled, target-adaptive agreement rule). All 10 project seeds were
used. CIC-DDoS2019's labels are used only to score already-fitted
predictions — never to train, tune, or threshold anything. The dataset
(431,371 rows) is small enough to hold in memory directly, so the
row-checkpointed streaming machinery built for CIC-ToN-IoT's 4.85M rows was
not needed; that is an engineering simplification, not a protocol change.

### Results

| Metric | CIC-ToN-IoT (frozen) | CIC-DDoS2019 (second) |
|---|---|---|
| **Deployed hybrid (selective_agreement_hybrid)** | | |
| MCC | -0.0002 ± 0.0006 (95% CI [-0.0006, 0.0000]) | **0.0000 ± 0.0000 exactly (95% CI [0.0000, 0.0000])** |
| Coverage | 0.694 ± 0.077 | 0.748 ± 0.079 |
| ROC-AUC (of the underlying mean probability) | 0.480 ± 0.041 | 0.855 ± 0.041 |
| ECE / Brier | 0.447 / 0.458 | 0.636 / 0.591 |
| **Adaptive DT branch (unselected)** | | |
| MCC | 0.160 ± 0.163 (95% CI [0.060, 0.249]) | 0.214 ± 0.107 (95% CI [0.142, 0.261]) |
| ROC-AUC | 0.575 ± 0.076 | 0.620 ± 0.053 |
| TPR / FPR | 0.372 / 0.222 | 0.305 / 0.072 |

Full table: `tables/confirmation_comparison_two_datasets.csv`. Figure:
`figures/second_confirmation_comparison.{png,pdf}`.

**Oracle-threshold diagnostic (seed 42, non-deployable upper bound, same
methodology as Task 3 Axis 4):**

| Branch | ROC-AUC | Frozen (actual) MCC | Oracle MCC | Recoverable |
|---|---|---|---|---|
| Decision Tree | 0.656 | 0.283 | 0.288 | 0.006 (negligible) |
| XGBoost | **0.918** | **0.000** | **0.660** | **0.660 (large)** |

Full detail: `tables/cicddos2019_oracle_threshold_seed42.json`.

### Does the main conclusion hold?

**Yes — reinforced, on a second, fully independent dataset, and even more
starkly than before.** The deployed selective-agreement hybrid's MCC is
exactly 0.0 in **all 10 of 10 seeds** on CIC-DDoS2019 (bootstrap CI is a
literal point mass at zero), a more total and uniform collapse than the
already-negative CIC-ToN-IoT result. The main claim of the paper — that
favorable internal/held-out-family results do not reliably predict
cross-domain performance, and that this specific frozen hybrid fails an
untouched external test — is not weakened by this second dataset; it is
corroborated by an independent capture, on a different attack category
(DDoS reflection/amplification vs. ToN-IoT's mixed IoT/network attacks),
processed through an independently maintained pipeline.

**The specific mechanistic culprit also reproduces almost exactly.** The
XGBoost branch's unlabeled, target-adaptive threshold (median + 6×IQR)
exceeds the valid [0, 1] probability range in **10 of 10 seeds** on
CIC-DDoS2019 (vs. 9 of 10 on CIC-ToN-IoT), which was already identified as
the mechanistic explanation for the confirmation failure. This is the
strongest single piece of new evidence in this addendum: the same brittle
threshold heuristic breaks the same way on two independent external
datasets.

**One genuine, honestly-reported difference: this failure is not primarily
a ranking failure on CIC-DDoS2019.** On CIC-ToN-IoT, ROC-AUC hovered near
chance (0.48–0.58), so there was little signal to recover even in
principle. On CIC-DDoS2019, the XGBoost branch's ROC-AUC is 0.918 —
genuinely strong — and the oracle-threshold diagnostic shows 0.660 of MCC
was recoverable by re-thresholding with the labels the frozen protocol is
not allowed to see. This means CIC-DDoS2019's failure is more purely a
threshold-selection failure than CIC-ToN-IoT's, which was closer to a
ranking failure. Both datasets end at the same operational outcome (a
useless deployed decision rule), but for a more precisely diagnosable
reason on CIC-DDoS2019, which if anything strengthens the paper's point
that unlabeled adaptive thresholding is the weak link, not the underlying
classifiers' discriminative capacity.

## Task 2: Robustness

### Method

Two independent techniques, both reusing only already-computed,
already-checksummed artifacts (`artifacts/redesign_v3/results/*.csv`,
`artifacts/redesign_v3/results/predictions/*_seed_42.npz`). No model was
retrained.

1. **Seed-level bootstrap 95% CIs** — identical methodology to the
   original `benchmark.redesign_analysis.bootstrap_intervals` (10 seeds
   resampled with replacement, 20,000 repetitions, `rng=default_rng(42)`),
   extended to three seed-level experiments it did not originally cover:
   the matched split-strategy control (Table II), the held-out-family
   study (Table IV), and the frozen confirmation domain (Table VI).
2. **Calibration sensitivity** — using the exact adapt/eval split already
   used by `recalibration_experiment` (same seed, same 2,000-row
   adaptation fold), (a) swept the applied temperature at
   {0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0}× the fitted optimum, and (b)
   refit the temperature on 5 independently-sampled adaptation folds at
   each of {500, 1000, 2000, 4000} rows to quantify sensitivity to
   adaptation-fold size.

Script: `benchmark/robustness_extension.py`. Outputs:
`tables/robustness_bootstrap_extended.csv`,
`tables/temperature_sensitivity_seed42.csv`,
`tables/adaptation_fold_sensitivity_seed42.csv` (+ `_raw.csv`),
`figures/robustness_bootstrap_ci.{png,pdf}`,
`manifests/robustness_extension_manifest.json`.

### Headline findings

**Table II (split-strategy MCC) is robust to seed resampling.** For all 8
mandatory models, the group-aware and random-row 95% bootstrap CIs do not
overlap (e.g., Decision Tree: group-aware [0.185, 0.258] vs. random-row
[0.853, 0.863]; XGBoost: [0.402, 0.462] vs. [0.881, 0.887]). This is an
independent confirmation of the existing Wilcoxon-Holm result using a
different statistical technique, not a re-derivation of the same test.

**Table VI (frozen confirmation) — the negative result is confirmed, with
one added nuance.** The actual, deployed selective-agreement hybrid's MCC
95% CI is [-0.0006, 0.0000] — a near-point-mass at exactly zero. Bootstrap
resampling across seeds does not move this conclusion at all. The
unselected decision-tree branch considered alone has a CI of
[0.060, 0.249], which excludes zero: a weak but seed-consistent non-chance
signal exists at the branch level, and the selective-agreement mechanism
removes even that. This sharpens, rather than weakens, the paper's
confirmation-failure claim.

**Table IV (held-out-family) MCCs are comfortably above zero for all 5
families and both models**, consistent with the manuscript's claim that
omitted attack types can remain detectable within-collection. The
family_aware_rf vs. redesigned_meta_hybrid CIs overlap for all 5 families,
consistent with the existing (non-significant after Holm correction)
pairwise comparison.

**Temperature scaling's MCC-invariance is confirmed to floating-point
exactness** (`baseline_mcc - temperature_scaled_mcc == 0.0` for all 10
model/domain pairs) — expected from the monotonicity of the transform, now
verified rather than assumed.

### Findings that complicate the story

**NLL-fit temperature is often not the ECE-minimizing temperature.** The
recalibration procedure (Guo et al., cited as [11]) selects T by minimizing
negative log-likelihood on the adaptation fold, which is standard practice
but is not the same objective as ECE. The sensitivity sweep shows that for
9 of 10 model/domain pairs, applying **half** the NLL-fitted temperature
would have produced **lower** ECE than the fitted optimum — sometimes
substantially (Decision Tree / CSE-CIC-IDS2018: ECE 0.167 → 0.020, an 88%
reduction at 0.5×). The one exception (XGBoost / CSE-CIC-IDS2018) moves the
other way, +5.0% at 0.5×, so this is a strong but not universal pattern. This is a genuine limitation of NLL-based temperature
fitting, not an implementation defect, and is disclosed here rather than
smoothed over. It does not change any reported MCC (temperature scaling
cannot, by construction) but it does mean the reported post-hoc ECE
reduction is conservative relative to what a directly ECE-tuned temperature
could achieve.

**One adaptation-fold draw produced a highly unstable temperature fit.**
For Random Forest / CIC-UNSW-NB15, one of five independent 1,000-row
adaptation draws fit T=2,460 (vs. 45–120
for the other four draws and the other fold sizes), and the search had to
be widened from the project's existing 500-point grid ceiling to 5,000 to
let it converge rather than hit an artificial wall. Despite this parameter
instability, the resulting evaluation-fold ECE stayed in a narrow band
(0.175–0.197) across all five draws — the calibration *outcome* is stable
even when the fitted *parameter* is not, because a sigmoid saturates
similarly for any sufficiently large T. This is reported because it is a
real fragility in the temperature-fitting step for at least one
model/domain pair, not because it changes any manuscript conclusion.

### Seed-count extension (10 → 20/30): assessed, not run

The task allowed increasing seed count "if feasible." Feasibility was
assessed using the project's own recorded per-model training times
(`tables/computational_efficiency.csv`): retraining all 8 mandatory models
on both split arms for 10 additional seeds (matched-split control alone)
is estimated at roughly 3.5–4 hours of sequential CPU time, dominated by
the three slow models (Family-aware RF, Tabular ResNet, Meta hybrid, each
~150–190s/seed/arm). Extending the main 3-domain benchmark and the
held-out-family study would add comparable time, and extending the frozen
confirmation experiment would be substantially more expensive still, since
it requires the checkpointed, row-by-row processing of 4,847,499 flows per
seed that the original protocol already treats as a memory-limited,
resumable, multi-phase job.

Given that (a) seed-level bootstrap CIs above already quantify seed-to-seed
uncertainty at negligible compute cost and show clean separation for every
headline claim, and (b) a multi-hour retraining commitment is a resource
decision the task instructions did not pre-authorize, this was not run
silently. It remains available on request; the exact commands are in
[Reproduction](#reproduction).

## Task 3: Controlled ablations

### Method

Four axes, each changing exactly one protocol component while holding
data, hyperparameters, seeds, and models identical. Script:
`benchmark/ablation_study.py`. Figure:
`figures/ablation_study_summary.{png,pdf}`.

| Axis | Without | With | New computation? |
|---|---|---|---|
| 1. Split strategy | Group-aware split | Random-row split | No — reuses Table II data |
| 2. Family exclusion | All families in training (internal benchmark) | One family held out | No — reuses Tables III/IV data |
| 3. Calibration | Raw probabilities | Temperature-scaled | No — reuses `recalibration_seed42.csv` |
| 4. Frozen confirmation threshold | Oracle (label-informed) threshold | Actual frozen (unlabeled) threshold | Yes — one new diagnostic, seed 42 only |
| 5. Semantic alignment | — | — | **Not executed; see below** |

### Results

**Axis 1 — split strategy is the largest single driver of MCC among the
four axes tested.** Random-row minus group-aware MCC deltas range from
+0.264 (CatBoost) to +0.821 (Family-aware RF) across all 8 models — every
model moves in the same direction, and the smallest delta (CatBoost,
+0.264) is still larger than the entire "with-calibration" or
"frozen-confirmation" deltas seen in axes 3–4.

**Axis 2 — family exclusion increases MCC for both models tested**, by
+0.594 (Family-aware RF: 0.095 → 0.690) and +0.370 (Meta hybrid:
0.288 → 0.658). This is *not* a contradiction of the manuscript: "without
exclusion" here means the standard internal group-aware benchmark
(Table III, all families present in training), and "with exclusion" pools
the 5 held-out-family results (Table IV). The held-out-family protocol
evaluates a smaller, differently-composed test slice (only flows from the
omitted family plus a matched benign pool) than the internal benchmark's
full held-out partition, so the two are not measuring identical test
distributions — the honest reading is that omitted-family detection is
strong *within this evaluation design*, exactly as the manuscript already
states, and this ablation should not be read as "exclusion helps
generalization" more broadly than that.

**Axis 3 — calibration changes ECE/Brier but never MCC**, confirmed
exactly (see Task 2). ECE moves in both directions depending on
model/domain (see the NLL-vs-ECE finding above); it is not a uniform
improvement.

**Axis 4 — for the decision-tree confirmation branch, the frozen
(unlabeled) threshold and the oracle (label-informed) threshold achieve
the same MCC** (0.0388 vs. 0.0388, recoverable = 0.00002) at seed 42. This
is a mechanistically important result: it means the seed-42 confirmation
failure for this branch is **not** a threshold-selection problem that an
oracle threshold could have fixed — it is a ranking problem (ROC-AUC at
seed 42 is 0.514, indistinguishable from chance). The frozen protocol did
not cost this branch any recoverable performance; the branch simply had
no separable signal to threshold at seed 42. (The XGBoost branch, not
part of this axis's "with/without frozen protocol" comparison because its
frozen threshold of 1.566 exceeds the valid probability range and
therefore cannot be swept, shows oracle MCC 0.164 at ROC-AUC 0.494 — also
near-chance ranking, reported in
`tables/confirmation_oracle_threshold_seed42.csv` for transparency even
though it is not part of the formal with/without comparison.)

**Axis 5 — semantic feature-schema alignment — was not executed.**
Removing it does not satisfy "one component changed, everything else
identical": every domain's raw export has a different, non-comparable
column set and dimensionality, so there is no well-defined fixed-length
input a frozen model could score without semantic alignment. The only way
to test "without alignment" is to train a new model on an ad hoc,
domain-specific feature space — a new modeling exercise, not a controlled
ablation, and squarely the kind of uncontrolled redesign the task
instructions rule out ("do NOT invent a new algorithm... do NOT redesign
the paper"). This is recorded explicitly rather than silently omitted.

## Traceability

| Output | Script | Inputs (pre-existing, unmodified) |
|---|---|---|
| `data/cic_ddos2019_second_confirmation_aligned/`, `manifests/cicddos2019_second_confirmation_dataset.json` | `benchmark/second_confirmation_cicddos2019.py::build_aligned_cache` | `data/raw/cic_ddos2019_staging/*.parquet` (downloaded from Kaggle `dhoogla/cicddos2019`), `benchmark/redesign_confirmation.py::column_mapping` (imported, unmodified) |
| `results/cicddos2019_second_confirmation_ten_seed_runs.csv`, `tables/cicddos2019_second_confirmation_mean_std.csv`, `results/predictions/cicddos2019_second_confirmation_seed42.npz` | `benchmark/second_confirmation_cicddos2019.py::run` | the aligned cache above; `results/group_tuning_*.json` (frozen hyperparameters, unmodified); development-domain data (`cicids2017`, `cse_cic_ids2018`, `cic_unsw_nb15`) |
| `tables/cicddos2019_oracle_threshold_seed42.json` | `benchmark/second_confirmation_cicddos2019.py::run` (seed-42 branch) | `results/cicddos2019_second_confirmation_ten_seed_runs.csv` |
| `tables/cicddos2019_second_confirmation_bootstrap_ci.csv` | ad hoc, same methodology as `extended_bootstrap_intervals` | `results/cicddos2019_second_confirmation_ten_seed_runs.csv` |
| `tables/confirmation_comparison_two_datasets.csv`, `figures/second_confirmation_comparison.{png,pdf}` | `benchmark/second_confirmation_figure.py` | `tables/confirmation_mean_std.csv`, `tables/cicddos2019_second_confirmation_mean_std.csv`, the two bootstrap CI tables above |
| `tables/robustness_bootstrap_extended.csv` | `benchmark/robustness_extension.py::extended_bootstrap_intervals` | `results/matched_split_combined_metrics.csv`, `results/unknown_family_ten_seed_runs.csv`, `results/confirmation_ten_seed_runs.csv` |
| `tables/temperature_sensitivity_seed42.csv` | `benchmark/robustness_extension.py::temperature_sensitivity` | `results/predictions/{model}_{domain}_seed_42.npz` |
| `tables/adaptation_fold_sensitivity_seed42*.csv` | `benchmark/robustness_extension.py::adaptation_fold_sensitivity` | `results/predictions/{model}_{domain}_seed_42.npz` |
| `tables/ablation_axis1_split_strategy.csv` | `benchmark/ablation_study.py::axis_split_strategy` | `results/grouped_ten_seed_runs.csv`, `results/matched_random_row_ten_seed_runs.csv`, `results/matched_random_row_extended_ten_seed_runs.csv` |
| `tables/ablation_axis2_family_exclusion.csv` | `benchmark/ablation_study.py::axis_family_exclusion` | `results/grouped_ten_seed_runs.csv`, `results/unknown_family_ten_seed_runs.csv` |
| `tables/ablation_axis3_calibration.csv` | `benchmark/ablation_study.py::axis_calibration` | `tables/recalibration_seed42.csv` |
| `tables/ablation_axis4_frozen_confirmation.csv`, `tables/confirmation_oracle_threshold_seed42.csv` | `benchmark/ablation_study.py::axis_frozen_confirmation` | `results/predictions/confirmation_seed42.npz`, `results/confirmation_ten_seed_runs.csv` |
| `figures/ablation_study_summary.{png,pdf}`, `figures/robustness_bootstrap_ci.{png,pdf}` | `benchmark/ablation_and_robustness_figures.py` | the tables above |
| `manifests/robustness_extension_manifest.json`, `manifests/ablation_study_manifest.json` | same scripts | — |

## Reproduction

```bash
# Task 1 (requires data/raw/cic_ddos2019_staging/*.parquet -- see below)
.venv/Scripts/python.exe -m benchmark.second_confirmation_cicddos2019
.venv/Scripts/python.exe -m benchmark.second_confirmation_figure

# Task 2
.venv/Scripts/python.exe -m benchmark.robustness_extension

# Task 3
.venv/Scripts/python.exe -m benchmark.ablation_study
.venv/Scripts/python.exe -m benchmark.ablation_and_robustness_figures
```

To re-download the CIC-DDoS2019 source data: obtain a Kaggle API token
(kaggle.com → Settings → API → Create New Token), save it to
`~/.kaggle/access_token`, then:

```bash
.venv/Scripts/python.exe -c "import kaggle; kaggle.api.authenticate(); \
  kaggle.api.dataset_download_files('dhoogla/cicddos2019', \
  path='data/raw/cic_ddos2019_staging', unzip=True)"
```

To extend seed count (not run in this addendum; see feasibility note
above), the existing reproduction commands in `README.md` accept
additional seeds via `benchmark/redesign_config.yaml`'s `study.seeds` list
— append new seed values and re-run the relevant stage(s)
(`redesign_supervised`, `redesign_hybrid`, `final_split_comparison[_extended]`,
`redesign_unknown_family`, or `redesign_confirmation`) for just the new
seeds, then re-run `benchmark.redesign_analysis` and this addendum's
scripts to refresh summaries.
