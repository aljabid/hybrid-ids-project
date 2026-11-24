import numpy as np
from collections import deque
import yaml
from utils.logger import get_logger
logger = get_logger(__name__)
class AdaptiveThreshold:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        cfg = config['adaptive_threshold']
        self.window_size       = cfg['window_size']
        self.min_threshold     = cfg['min_threshold']
        self.max_threshold     = cfg['max_threshold']
        self.update_rate       = cfg['update_rate']
        self.current_threshold = config['hybrid']['confidence_threshold']
        self._window   = deque(maxlen=self.window_size)
        self._history  = []
        logger.info(f"AdaptiveThreshold initialized | "
                    f"window={self.window_size} | "
                    f"init_threshold={self.current_threshold} | "
                    f"range=[{self.min_threshold}, {self.max_threshold}]")
    def update(self, confidence_score: float) -> float:
        self._window.append(confidence_score)
        if len(self._window) >= self.window_size:
            window_mean       = float(np.mean(self._window))
            new_threshold     = (window_mean * self.update_rate +
                                  self.current_threshold * (1 - self.update_rate))
            self.current_threshold = float(np.clip(
                new_threshold, self.min_threshold, self.max_threshold))
        self._history.append({
            "step":      len(self._history),
            "threshold": round(self.current_threshold, 4),
            "score":     round(float(confidence_score), 4),
            "window_mean": round(float(np.mean(self._window)), 4)
                           if len(self._window) > 0 else None
        })
        return self.current_threshold
    def get_threshold(self) -> float:
        return self.current_threshold
    def get_history(self) -> list:
        return self._history
    def reset(self):
        self._window.clear()
        self._history.clear()
        logger.info("AdaptiveThreshold reset")
def plot_threshold_history(history, output_path):
    import matplotlib.pyplot as plt
    steps      = [h['step'] for h in history]
    thresholds = [h['threshold'] for h in history]
    scores     = [h['score'] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(steps, scores, alpha=0.4, color='steelblue',
                 label='Confidence Score', linewidth=0.8)
    axes[0].plot(steps, thresholds, color='red', linewidth=2,
                 label='Adaptive Threshold')
    axes[0].set_ylabel('Score / Threshold', fontweight='bold')
    axes[0].set_title('Adaptive Confidence Threshold Over Time',
                      fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, thresholds, color='red', linewidth=2)
    axes[1].set_xlabel('Flow Index', fontweight='bold')
    axes[1].set_ylabel('Threshold', fontweight='bold')
    axes[1].set_title('Threshold Evolution', fontweight='bold')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Adaptive threshold history saved → {output_path}")
