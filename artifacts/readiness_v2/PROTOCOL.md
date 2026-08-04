# Frozen Scientific Protocol — Manuscript Readiness v2

## Primary hypothesis

A validation-selected, calibrated hybrid of the strongest supervised backbone
and a normal-only anomaly detector improves cross-domain and unseen-family MCC
and PR-AUC over the calibrated Random Forest without increasing FPR by more
than 0.02.

## Selection and test separation

- Model, scaler, calibrator, anomaly detector, fusion rule, and thresholds are
  selected using CICIDS2017 training and validation partitions only.
- CICIDS2017 internal test, CSE-CIC-IDS2018, and CIC-UNSW-NB15 are frozen test
  domains. Their labels cannot influence selection.
- External scores may be inspected only for final reporting after the selected
  configuration is written to a freeze manifest.
- Every final comparison uses the same ten paired seeds.

## Primary outcome

Matthews correlation coefficient (MCC). PR-AUC, TPR, FPR, ECE, Brier score,
latency, throughput, and model size are secondary outcomes.

## Positive-method gates

1. External MCC improves by at least 0.02 over calibrated Random Forest.
2. External TPR improves by at least 0.05.
3. At least 60% of eligible held-out attack families improve.
4. Absolute FPR increase is no greater than 0.02.
5. The primary paired comparison has Holm-adjusted p <= 0.05.

Failure to meet these gates produces a not-ready verdict for a positive
superiority manuscript. No external-label retuning is permitted.

## Real-data policy

Synthetic, generated, mock, or silently substituted flows are prohibited.
Unavailable metadata or incompatible domains must be documented and excluded.

## Computational bound

The repeated MLP experiment uses a fixed 20-iteration upper bound, mini-batches
of 2,048, and early stopping, applied identically to all ten seeds. These
resource settings were declared before
completion of the final repeated experiment after uncapped and 60-iteration
full-data attempts proved computationally infeasible. All MLP rows from those
attempts are discarded and rerun under this final bound.

## Temporal decision

The supplied CICIDS2017 CSVs contain no timestamps or source identifiers.
CSE-CIC-IDS2018 contains timestamps but no source IP. Valid per-source temporal
aggregation cannot be reconstructed consistently across domains. The temporal
component is removed from the validated contribution unless metadata-preserving
official flows are later acquired.
