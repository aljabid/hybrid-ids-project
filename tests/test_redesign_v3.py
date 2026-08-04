import json
from pathlib import Path

import numpy as np

from benchmark.metrics import binary_metrics
from benchmark.redesign_confirmation import streaming_binary_metrics
from benchmark.redesign_selective import predict_selective


ROOT = Path("artifacts/redesign_v3")


def test_streaming_confirmation_metrics_match_exact_metrics():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 20_000, dtype=np.int8)
    probability = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.3, len(y)), 0, 1)
    prediction = probability >= 0.5
    exact = binary_metrics(y, prediction, probability)
    streaming = streaming_binary_metrics(y, prediction, probability)
    for metric in (
        "accuracy", "precision", "recall", "f1", "mcc", "tpr", "fpr",
        "ece", "brier", "negative_log_likelihood",
    ):
        assert np.isclose(streaming[metric], exact[metric], atol=1e-12)
    assert abs(streaming["roc_auc"] - exact["roc_auc"]) < 1e-4
    assert abs(streaming["pr_auc"] - exact["pr_auc"]) < 1e-4


def test_selective_prediction_only_covers_branch_agreement():
    decision_tree = np.array([0.01, 0.99, 0.01, 0.99] * 100)
    xgboost = np.array([0.01, 0.99, 0.99, 0.01] * 100)
    result = predict_selective(decision_tree, xgboost)
    dt = result["dt_prediction"]
    xgb = result["xgb_prediction"]
    assert np.array_equal(result["covered"], dt == xgb)
    assert np.array_equal(
        result["selective_prediction"], dt[result["covered"]]
    )


def test_confirmation_was_frozen_and_all_seeds_completed():
    freeze = json.loads(
        (ROOT / "manifests/final_method_freeze.json").read_text()
    )
    execution = json.loads(
        (ROOT / "manifests/confirmation_execution.json").read_text()
    )
    assert freeze["status"] == "FINAL_METHOD_FROZEN_BEFORE_CONFIRMATION_DOWNLOAD"
    assert freeze["confirmation_downloaded"] is False
    assert execution["method_frozen_before_access"] is True
    assert execution["labels_used_for_revision"] is False
    assert execution["seeds_completed"] == [7, 19, 31, 42, 53, 67, 79, 97, 109, 127]
    assert execution["rows"] == 4_847_499


def test_claims_audit_forbids_superiority_claim():
    audit = json.loads((ROOT / "manifests/claims_audit.json").read_text())
    assert audit["real_data_only"] is True
    assert audit["mock_or_synthetic_rows"] == 0
    assert audit["consistent_superiority_claim_allowed"] is False
    assert audit["confirmation_labels_used_for_revision"] is False
