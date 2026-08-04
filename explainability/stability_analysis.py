   
                                    

                                                                   
                                                                
                                

         
                                            
                                             
   

import os
import sys
import json
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')

ATTR_CICIDS = "results/metrics/shap_class_attribution_cicids.json"
ATTR_IDS18 = "results/metrics/shap_class_attribution_ids2018.json"
FIG_OUT = "results/figures/figure3_shap_stability.png"
METRICS_OUT = "results/metrics/shap_stability_metrics.json"


def load_attributions():
       
                                            
                                                       
       
    if os.path.exists(ATTR_CICIDS) and os.path.exists(ATTR_IDS18):
        with open(ATTR_CICIDS) as f:
            attr_cicids = json.load(f)
        with open(ATTR_IDS18) as f:
            attr_ids18 = json.load(f)
        logger.info("Loaded SHAP attributions from disk")
        return attr_cicids, attr_ids18
    else:
        logger.warning("SHAP attribution files not found — using demo data")
        logger.warning("Run explainability/shap_analysis.py first")
        features = [
            'byte_count', 'bytes_per_packet', 'log_byte_count',
            'ack_flag_count', 'flag_sum', 'ack_ratio',
            'fin_ratio', 'syn_ratio', 'packet_rate', 'log_packet_rate'
        ]
        np.random.seed(42)
        base = np.abs(np.random.randn(len(features))) * 0.1
        attr_c = {f: round(float(v), 6)
                  for f, v in zip(features, base + np.abs(np.random.randn(len(features))) * 0.02)}
        attr_i = {f: round(float(v), 6)
                  for f, v in zip(features, base * 0.7 + np.abs(np.random.randn(len(features))) * 0.05)}
                   
        tc = sum(attr_c.values());
        attr_c = {k: round(v / tc, 6) for k, v in attr_c.items()}
        ti = sum(attr_i.values());
        attr_i = {k: round(v / ti, 6) for k, v in attr_i.items()}
        return attr_c, attr_i


def rank_features(attr_dict):
                                                               
    sorted_feats = sorted(attr_dict.items(), key=lambda x: x[1], reverse=True)
    return {feat: rank + 1 for rank, (feat, _) in enumerate(sorted_feats)}


def run():
    attr_cicids, attr_ids18 = load_attributions()

                                       
    common_features = sorted(set(attr_cicids.keys()) & set(attr_ids18.keys()))
    logger.info(f"Common features: {len(common_features)}")

                                               
    tc = sum(attr_cicids[f] for f in common_features)
    ti = sum(attr_ids18[f] for f in common_features)
    norm_cicids = {f: attr_cicids[f] / tc for f in common_features}
    norm_ids18 = {f: attr_ids18[f] / ti for f in common_features}

                                   
    sorted_by_cicids = sorted(common_features,
                              key=lambda f: norm_cicids[f], reverse=True)

    vals_cicids = np.array([norm_cicids[f] for f in sorted_by_cicids])
    vals_ids18 = np.array([norm_ids18[f] for f in sorted_by_cicids])
    delta = vals_ids18 - vals_cicids

                      
    rank_c = rank_features(norm_cicids)
    rank_i = rank_features(norm_ids18)
    rc = [rank_c[f] for f in common_features]
    ri = [rank_i[f] for f in common_features]
    spearman_r, spearman_p = scipy_stats.spearmanr(rc, ri)
    logger.info(f"Spearman rank correlation: r={spearman_r:.4f}  p={spearman_p:.4f}")

                                                               
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        'SHAP Feature Importance Stability: CICIDS2017 → CIC-IDS2018\n'
        f'Spearman Rank Correlation: r={spearman_r:.3f}  '
        f'(1.0 = perfectly stable, 0.0 = no stability)',
        fontsize=13, fontweight='bold'
    )

    n = len(sorted_by_cicids)
    x = np.arange(n)
    w = 0.38

                                 
    ax = axes[0]
    ax.barh(x + w / 2, vals_cicids, w, color='#2196F3', alpha=0.85,
            label='CICIDS2017', edgecolor='white')
    ax.barh(x - w / 2, vals_ids18, w, color='#FF9800', alpha=0.85,
            label='IDS2018', edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(sorted_by_cicids, fontsize=9)
    ax.set_xlabel('Normalized Mean |SHAP|', fontweight='bold')
    ax.set_title('A — Feature Importance Comparison', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

                            
    ax = axes[1]
    rank_cicids_arr = np.array([rank_c[f] for f in common_features])
    rank_ids18_arr = np.array([rank_i[f] for f in common_features])
    colors_pt = plt.cm.tab10(np.linspace(0, 1, len(common_features)))

    ax.plot([1, n], [1, n], 'k--', alpha=0.4, linewidth=1.5,
            label='Perfect stability')
    for i, feat in enumerate(common_features):
        ax.scatter(rank_c[feat], rank_i[feat],
                   color=colors_pt[i], s=80, zorder=3)
        ax.annotate(feat,
                    (rank_c[feat], rank_i[feat]),
                    textcoords='offset points',
                    xytext=(5, 3), fontsize=7)

    ax.set_xlabel('CICIDS2017 Rank', fontweight='bold')
    ax.set_ylabel('CIC-IDS2018 Rank', fontweight='bold')
    ax.set_title('B — Rank Stability\n(diagonal = unchanged rank)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)

                          
    ax.text(0.05, 0.95, f'Spearman r = {spearman_r:.3f}',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

                          
    ax = axes[2]
    bar_colors = ['#F44336' if d > 0 else '#2196F3' for d in delta]
    bars = ax.barh(sorted_by_cicids, delta, color=bar_colors, alpha=0.85,
                   edgecolor='white')
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_xlabel('Δ Importance (IDS2018 − CICIDS2017)', fontweight='bold')
    ax.set_title('C — Importance Shift\n'
                 '(red = gained importance, blue = lost importance)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

                      
    for bar, val in zip(bars, delta):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.4f}', va='center', fontsize=7,
                ha='left' if val >= 0 else 'right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP stability figure saved → {FIG_OUT}")

                       
    metrics = {
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 6),
        "stability_verdict": ("stable" if spearman_r > 0.7
                              else "moderately stable" if spearman_r > 0.4
        else "unstable"),
        "most_stable_features": [
            f for f in sorted_by_cicids
            if abs(norm_ids18[f] - norm_cicids[f]) < 0.02
        ],
        "most_shifted_features": [
            {"feature": f,
             "cicids_importance": round(norm_cicids[f], 4),
             "ids2018_importance": round(norm_ids18[f], 4),
             "delta": round(float(norm_ids18[f] - norm_cicids[f]), 4)}
            for f in sorted(common_features,
                            key=lambda x: abs(norm_ids18[x] - norm_cicids[x]),
                            reverse=True)[:3]
        ]
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Stability metrics saved → {METRICS_OUT}")

    logger.info(f"\nStability verdict: {metrics['stability_verdict']}")
    logger.info(f"Most shifted features:")
    for item in metrics['most_shifted_features']:
        logger.info(f"  {item['feature']}: delta={item['delta']:+.4f}")

    return metrics


if __name__ == "__main__":
    run()