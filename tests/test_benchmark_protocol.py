import numpy as np
from benchmark.hybrid_experiments import adaptive_router, fixed_router
from benchmark.metrics import binary_metrics, expected_calibration_error
def test_binary_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.01, 0.1, 0.9, 0.99])
    metrics = binary_metrics(y, probability >= 0.5, probability)
    assert metrics["accuracy"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["tpr"] == 1.0
    assert metrics["fpr"] == 0.0
def test_ece_is_bounded():
    ece, bins = expected_calibration_error(
        np.array([0, 1]), np.array([0.1, 0.9])
    )
    assert 0 <= ece <= 1
    assert bins
def test_fixed_router_sends_confident_attack_to_classifier():
    prediction, score, route = fixed_router(
        np.array([0.99, 0.1]),
        np.array([0.2, 0.99]),
        confidence_threshold=0.9,
        anomaly_threshold=0.95,
    )
    assert prediction.tolist() == [1, 1]
    assert route.tolist() == [True, False]
    assert score.tolist() == [0.99, 0.99]
def test_adaptive_threshold_changes_decision_path():
    probability = np.full(60, 0.92)
    anomaly = np.zeros(60)
    _, _, route, thresholds = adaptive_router(
        probability,
        anomaly,
        initial_threshold=0.85,
        anomaly_threshold=0.95,
        window_size=5,
        update_rate=0.5,
    )
    assert route.any()
    assert thresholds[-1] > thresholds[0]
