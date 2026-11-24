import os
import sys
import pytest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybrid.adaptive_threshold import AdaptiveThreshold
class TestAdaptiveThreshold:
    def test_initialization(self):
        at = AdaptiveThreshold()
        assert 0.0 < at.current_threshold <= 1.0
        assert at.window_size > 0
    def test_update_returns_float(self):
        at  = AdaptiveThreshold()
        val = at.update(0.9)
        assert isinstance(val, float)
    def test_threshold_stays_in_bounds(self):
        at = AdaptiveThreshold()
        for score in np.random.rand(200):
            thresh = at.update(float(score))
            assert at.min_threshold <= thresh <= at.max_threshold
    def test_high_confidence_raises_threshold(self):
        at = AdaptiveThreshold()
        init_thresh = at.current_threshold
        for _ in range(100):
            at.update(1.0)
        assert at.current_threshold >= init_thresh
    def test_low_confidence_lowers_threshold(self):
        at = AdaptiveThreshold()
        init_thresh = at.current_threshold
        for _ in range(200):
            at.update(0.0)
        assert at.current_threshold <= init_thresh
    def test_history_grows(self):
        at = AdaptiveThreshold()
        for i in range(10):
            at.update(0.8)
        assert len(at.get_history()) == 10
    def test_reset(self):
        at = AdaptiveThreshold()
        for _ in range(20):
            at.update(0.9)
        at.reset()
        assert len(at.get_history()) == 0
from hybrid.temporal_aggregator import TemporalAggregator
class TestTemporalAggregator:
    def test_initialization(self):
        ta = TemporalAggregator(ae_threshold=0.05)
        assert ta.window_size > 0
        assert ta.alert_threshold > ta.ae_threshold
    def test_update_returns_dict(self):
        ta     = TemporalAggregator(ae_threshold=0.05)
        result = ta.update("192.168.1.1", 0.01)
        assert 'alert' in result
        assert 'window_mean' in result
        assert 'flow_idx' in result
    def test_no_alert_for_normal_errors(self):
        ta = TemporalAggregator(ae_threshold=0.05)
        for _ in range(20):
            result = ta.update("192.168.1.1", 0.001)
        assert not result['alert']
    def test_alert_fires_for_high_errors(self):
        ta = TemporalAggregator(ae_threshold=0.05)
        alert_fired = False
        for _ in range(20):
            result = ta.update("192.168.1.2", 999.0)
            if result['alert']:
                alert_fired = True
                break
        assert alert_fired
    def test_per_ip_isolation(self):
        ta = TemporalAggregator(ae_threshold=0.05)
        for _ in range(15):
            ta.update("192.168.1.1", 0.001)
        for _ in range(15):
            ta.update("192.168.1.2", 999.0)
        alerts = ta.get_alerts()
        alerted_ips = [a['source_ip'] for a in alerts]
        assert "192.168.1.2" in alerted_ips
        assert "192.168.1.1" not in alerted_ips
    def test_alert_latency_recorded(self):
        ta = TemporalAggregator(ae_threshold=0.05)
        for _ in range(20):
            ta.update("10.0.0.1", 999.0)
        latencies = ta.get_alert_latencies()
        assert len(latencies) == 1
        assert latencies[0] >= 1
    def test_reset_clears_state(self):
        ta = TemporalAggregator(ae_threshold=0.05)
        for _ in range(20):
            ta.update("192.168.1.1", 999.0)
        ta.reset()
        assert len(ta.get_alerts()) == 0
from utils.metrics_utils import compute_ece
class TestECE:
    def test_perfect_calibration(self):
        y_true = np.array([1]*50 + [0]*50)
        y_pred = np.array([1]*50 + [0]*50)
        conf   = np.array([1.0]*50 + [0.0]*50)
        ece, _ = compute_ece(y_true, y_pred, conf)
        assert ece < 0.15
    def test_worst_calibration(self):
        y_true = np.array([0]*100)
        y_pred = np.array([1]*100)
        conf   = np.array([0.99]*100)
        ece, _ = compute_ece(y_true, y_pred, conf)
        assert ece > 0.5
    def test_ece_in_unit_range(self):
        for _ in range(10):
            n      = 200
            y_true = np.random.randint(0, 2, n)
            y_pred = np.random.randint(0, 2, n)
            conf   = np.random.rand(n)
            ece, _ = compute_ece(y_true, y_pred, conf)
            assert 0.0 <= ece <= 1.0
    def test_bin_details_structure(self):
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        conf   = np.random.rand(100)
        _, bins = compute_ece(y_true, y_pred, conf)
        for b in bins:
            assert 'bin' in b
            assert 'accuracy' in b
            assert 'confidence' in b
            assert 'gap' in b
