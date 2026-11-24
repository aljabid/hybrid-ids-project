import numpy as np
from collections import defaultdict, deque
import yaml
from utils.logger import get_logger
logger = get_logger(__name__)
class TemporalAggregator:
    def __init__(self, ae_threshold, config_path="config/config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        cfg = config['temporal_aggregator']
        self.window_size     = cfg['window_size']
        self.alert_multiplier = cfg['alert_multiplier']
        self.ae_threshold    = ae_threshold
        self.alert_threshold = ae_threshold * self.alert_multiplier
        self._ip_windows    = defaultdict(lambda: deque(maxlen=self.window_size))
        self._ip_flow_count = defaultdict(int)
        self._ip_alert_idx  = {}
        self._alerts        = []
        logger.info(f"TemporalAggregator | window={self.window_size} | "
                    f"alert_threshold={self.alert_threshold:.6f} "
                    f"(ae_threshold × {self.alert_multiplier})")
    def update(self, source_ip: str, error: float) -> dict:
        self._ip_windows[source_ip].append(error)
        self._ip_flow_count[source_ip] += 1
        flow_idx = self._ip_flow_count[source_ip]
        window      = self._ip_windows[source_ip]
        window_mean = float(np.mean(window))
        alert       = window_mean > self.alert_threshold
        result = {
            "source_ip":    source_ip,
            "flow_idx":     flow_idx,
            "error":        round(error, 6),
            "window_mean":  round(window_mean, 6),
            "window_size":  len(window),
            "alert":        alert,
            "first_alert":  False
        }
        if alert and source_ip not in self._ip_alert_idx:
            self._ip_alert_idx[source_ip] = flow_idx
            result["first_alert"] = True
            self._alerts.append({
                "source_ip":     source_ip,
                "alert_at_flow": flow_idx,
                "window_mean":   round(window_mean, 6),
            })
            logger.info(f"TEMPORAL ALERT | IP={source_ip} | "
                        f"flow #{flow_idx} | mean_error={window_mean:.6f}")
        return result
    def get_alert_latencies(self) -> list:
        return [a['alert_at_flow'] for a in self._alerts]
    def get_mean_alert_latency(self) -> float:
        lats = self.get_alert_latencies()
        return float(np.mean(lats)) if lats else float('nan')
    def get_error_series(self, source_ip: str) -> list:
        return list(self._ip_windows.get(source_ip, []))
    def get_alerts(self) -> list:
        return self._alerts
    def reset(self):
        self._ip_windows.clear()
        self._ip_flow_count.clear()
        self._ip_alert_idx.clear()
        self._alerts.clear()
