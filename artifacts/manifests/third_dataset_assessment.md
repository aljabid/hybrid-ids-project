# Third-dataset alignment assessment

UNSW-NB15 was not present in the supplied repository and was therefore not used
in the executed benchmark. The current ten-feature schema depends on CICFlowMeter
semantics such as forward-packet length, flow packet rate, and TCP flag counts.
UNSW-NB15 exposes related but not fully equivalent bidirectional connection
statistics. Treating similarly named fields as exact matches, or filling absent
fields with zero, would create an invalid cross-domain comparison.

The benchmark therefore treats CSE-CIC-IDS2018 as the mandatory external domain.
UNSW-NB15 can be added only after:

1. acquiring the official dataset;
2. producing a reviewed semantic mapping for every retained feature;
3. regenerating all training-domain preprocessing without consulting its labels;
4. recording exact, derived, approximate, and unavailable mappings.

This is an explicit exclusion based on missing source data and feature semantics,
not a failed result hidden from reporting.

