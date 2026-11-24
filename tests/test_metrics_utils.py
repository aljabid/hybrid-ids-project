import os
import sys
import pytest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics_utils import (
    compute_accuracy, compute_f1, compute_tpr_fpr,
    full_metrics_dict
)
class TestMetrics:
    def test_perfect_accuracy(self):
        y = np.array([0, 1, 2, 3])
        assert compute_accuracy(y, y) == 1.0
    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert compute_accuracy(y_true, y_pred) == 0.0
    def test_tpr_fpr_binary(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        tpr, fpr = compute_tpr_fpr(y_true, y_pred)
        assert tpr == 0.5
        assert fpr == 0.5
    def test_perfect_tpr_zero_fpr(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        tpr, fpr = compute_tpr_fpr(y_true, y_pred)
        assert tpr == 1.0
        assert fpr == 0.0
    def test_full_metrics_dict_keys(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        m = full_metrics_dict(y_true, y_pred)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'tpr', 'fpr']:
            assert key in m
    def test_all_metrics_in_unit_range(self):
        for _ in range(10):
            n      = 100
            y_true = np.random.randint(0, 2, n)
            y_pred = np.random.randint(0, 2, n)
            m      = full_metrics_dict(y_true, y_pred)
            for key in ['accuracy', 'precision', 'recall', 'f1', 'tpr', 'fpr']:
                assert 0.0 <= m[key] <= 1.0, \
                    f"{key} = {m[key]} out of range"
