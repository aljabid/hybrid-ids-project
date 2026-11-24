import pandas as pd
import matplotlib.pyplot as plt
def run():
    cicids2017 = pd.read_csv("data/processed/cicids2017/cicids_aligned_features.csv")
    cicids2018 = pd.read_csv("data/processed/cic_ids2018/cic2018_aligned_features.csv")
    cicids2017['dataset'] = 'CICIDS2017'
    cicids2018['dataset'] = 'CICIDS2018'
    combined = pd.concat([cicids2017, cicids2018], axis=0)
    feature_cols = ['byte_count','bytes_per_packet','log_byte_count','ack_flag_count',
                    'flag_sum','ack_ratio','fin_ratio','syn_ratio','packet_rate','log_packet_rate']
    n_features = len(feature_cols)
    fig, axes = plt.subplots(nrows=2, ncols=(n_features // 2), figsize=(20,8))
    axes = axes.flatten()
    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        for dataset in ['CICIDS2017','CICIDS2018']:
            subset = combined[combined['dataset'] == dataset][feat]
            ax.hist(subset, bins=50, alpha=0.5, label=dataset)
        ax.set_title(feat)
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/feature_distribution_comparison.png", dpi=300)
    plt.close()
