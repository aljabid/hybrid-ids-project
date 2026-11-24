import os
import sys
import pytest
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder.autoencoder_model import Autoencoder
class TestAutoencoderArchitecture:
    def test_output_shape(self):
        ae = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        x  = torch.randn(8, 10)
        out = ae(x)
        assert out.shape == (8, 10)
    def test_output_range(self):
        ae  = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        x   = torch.randn(100, 10)
        out = ae(x)
        assert (out >= 0).all()
        assert (out <= 1).all()
    def test_reconstruction_error_shape(self):
        ae     = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        x      = torch.randn(20, 10)
        errors = ae.reconstruction_error(x)
        assert errors.shape == (20,)
    def test_reconstruction_error_nonnegative(self):
        ae     = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        x      = torch.rand(50, 10)
        errors = ae.reconstruction_error(x)
        assert (errors >= 0).all()
    def test_encoder_bottleneck_shape(self):
        ae = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        x  = torch.randn(5, 10)
        z  = ae.encode(x)
        assert z.shape == (5, 16)
    def test_normal_lower_error_than_anomaly(self):
        torch.manual_seed(42)
        ae        = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        normal_data = torch.zeros(200, 10) + torch.randn(200, 10) * 0.05
        for _ in range(100):
            optimizer.zero_grad()
            out  = ae(normal_data)
            loss = criterion(out, normal_data)
            loss.backward()
            optimizer.step()
        ae.eval()
        normal_test  = torch.zeros(50, 10) + torch.randn(50, 10) * 0.05
        anomaly_test = torch.ones(50, 10) * 5.0
        normal_errors  = ae.reconstruction_error(normal_test).mean().item()
        anomaly_errors = ae.reconstruction_error(anomaly_test).mean().item()
        assert anomaly_errors > normal_errors, \
            f"Anomaly error ({anomaly_errors:.4f}) should exceed " \
            f"normal error ({normal_errors:.4f})"
class TestThreshold:
    def test_threshold_above_mean(self):
        from utils.metrics_utils import compute_ece
        errors = np.random.exponential(0.01, 500)
        mu     = errors.mean()
        sigma  = errors.std()
        theta  = mu + 2 * sigma
        assert theta > mu
    def test_ece_range(self):
        from utils.metrics_utils import compute_ece
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        conf   = np.array([0.9, 0.9, 0.8, 0.8, 0.95, 0.95, 0.7, 0.7, 0.85, 0.85])
        ece, _ = compute_ece(y_true, y_pred, conf)
        assert 0.0 <= ece <= 1.0
