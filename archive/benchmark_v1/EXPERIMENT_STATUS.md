# Hybrid IDS Experimental Package Status

This is an experimental audit and artifact index, not a manuscript.

## Completed phases

1. **Data integrity and provenance** — duplicate removal, contradictory-row
   removal, immutable disjoint splits, SHA-256 manifests, and overlap audit.
2. **Supervised benchmark** — Decision Tree, Random Forest, XGBoost, LightGBM,
   CatBoost, and MLP under one fixed feature/scaling contract.
3. **Hyperparameter optimization** — training-domain-only randomized search
   with five-fold stratified CV and MCC scoring.
4. **Cross-domain evaluation** — untouched CICIDS2017 internal test and
   CSE-CIC-IDS2018 external test. UNSW-NB15 exclusion is documented.
5. **Ten-seed robustness** — paired seeds for every supervised and hybrid
   experiment, with raw predictions retained.
6. **Hybrid ablations** — classifier, autoencoder, fixed, confidence-routed,
   and adaptive variants.
7. **Unknown-attack evaluation** — leave-one-family-out experiments for the
   five largest eligible CICIDS2017 attack families across ten seeds.
8. **Statistics** — paired Wilcoxon tests, Holm correction, effect size, and
   win/tie/loss counts.
9. **Calibration** — ECE, Brier score, negative log-likelihood, and reliability
   diagrams for every supervised model.
10. **Explainability** — SHAP values and cross-domain top-feature rankings for
    all six supervised models.
11. **Efficiency** — training time, inference time, per-flow latency,
    throughput, peak process RSS, and serialized model size.
12. **Error and drift analyses** — confusion matrices, false-positive and
    false-negative summaries, Jensen–Shannon divergence, PSI, and range drift.
13. **Figures and tables** — vector PDF, 600-DPI PNG, CSV, and Excel workbook.
14. **Reproducibility** — isolated environment instructions, central YAML
    configuration, raw seed artifacts, manifests, tests, and one entry point.

## Scope limitation

Temporal aggregation is implemented in the legacy project but is not presented
as an experimentally validated ablation. The supplied aligned matrices do not
retain legitimate source identifiers and packet/flow ordering. Reusing the true
label as a proxy identifier, as the legacy evaluator did, would leak ground
truth. Temporal validation must wait for metadata-preserving preprocessing.

## Principal numerical findings

- Random Forest was the strongest training-CV backbone (`MCC = 0.9902`).
- Internal CICIDS2017 performance remained high for all tree models.
- All supervised models suffered catastrophic external attack recall on the
  supplied CSE-CIC-IDS2018 domain (`TPR ≈ 0.0006`).
- The current hybrid increased external TPR only to roughly `0.0011` and
  increased false positives; it did not outperform Random Forest.
- Leave-one-family-out results show that the current anomaly-routing mechanism
  usually decreases unknown-family recall. It cannot presently support a claim
  of improved unknown-attack detection.
- Packet rate, log packet rate, byte count, log byte count, and bytes per packet
  exhibit the largest CICIDS2017→CSE-CIC-IDS2018 distribution drift.

The experiment package is therefore reproducible and substantially more
rigorous than the legacy evaluation, but the present method requires redesign
before making superiority or unknown-attack claims.
