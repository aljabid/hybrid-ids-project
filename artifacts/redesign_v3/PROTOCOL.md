# Domain-robust Hybrid IDS redesign protocol (v3)

## Locked objective

Develop a semantically aligned, group-robust IDS whose conservative hybrid
component improves held-out collection/family detection without increasing the
false-positive rate by more than 0.02.

## Separation rules

- CICIDS2017 source files are collection groups. No feature-identical flow may
  cross training, validation, or internal-test partitions.
- Model, preprocessing, anomaly, calibration, gating, and thresholds are
  selected using CICIDS2017 training and validation groups.
- Previously observed CSE-CIC-IDS2018 and CIC-UNSW results are development
  diagnostics and cannot establish final generalization.
- A new confirmation dataset remains inaccessible until `method_freeze.json`
  is written. It will be evaluated once without label-driven revision.
- All publication experiments use real flows only and ten paired seeds.

## Feature policy

Use only explicitly mapped CICFlowMeter flow statistics with compatible units
and direction semantics. Exclude destination port, protocol, timestamp,
initial-window fields, duplicated columns, and unstable bulk aggregates.

## Temporal policy

No temporal claim is permitted because consistent source/session identifiers
are unavailable across the supplied datasets.

## Stopping rule

If validation gates fail, the method receives a not-ready result and the
untouched confirmation dataset is not opened merely to search for a favorable
outcome.
