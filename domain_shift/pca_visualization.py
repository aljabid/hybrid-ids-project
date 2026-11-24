import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import umap
MAX_SAMPLES = 20000
RANDOM_STATE = 42
def load_and_sample(path, label, n_samples=MAX_SAMPLES):
    df = pd.read_csv(path)
    df["dataset"] = label
    if len(df) > n_samples:
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            random_state=RANDOM_STATE
        )
    return df
def run():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    cicids2017 = load_and_sample(
        "data/processed/cicids2017/cicids_aligned_features.csv",
        "CICIDS2017"
    )
    cicids2018 = load_and_sample(
        "data/processed/cic_ids2018/cic2018_aligned_features.csv",
        "CICIDS2018"
    )
    combined = pd.concat([cicids2017, cicids2018], axis=0)
    feature_cols = [
        "byte_count",
        "bytes_per_packet",
        "log_byte_count",
        "ack_flag_count",
        "flag_sum",
        "ack_ratio",
        "fin_ratio",
        "syn_ratio",
        "packet_rate",
        "log_packet_rate"
    ]
    X = combined[feature_cols].values
    y = combined["dataset"].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    colors = {
        "CICIDS2017": "#2196F3",
        "CICIDS2018": "#FF9800"
    }
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    np.save("results/metrics/pca_variance.npy", var_exp)
    plt.figure(figsize=(8, 6))
    for ds in colors:
        mask = (y == ds)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=ds,
            alpha=0.6,
            color=colors[ds]
        )
    plt.title(
        "PCA Distribution Shift:\nCICIDS2017 vs CIC-IDS2018",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel(
        f"PC1 ({var_exp[0]*100:.1f}% variance)",
        fontsize=12,
        fontweight="bold"
    )
    plt.ylabel(
        f"PC2 ({var_exp[1]*100:.1f}% variance)",
        fontsize=12,
        fontweight="bold"
    )
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(
        "results/figures/pca_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=30,
        max_iter=500,
        init="pca",
        learning_rate="auto"
    )
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for ds in colors:
        mask = (y == ds)
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            label=ds,
            alpha=0.6,
            color=colors[ds]
        )
    plt.title(
        "t-SNE Distribution Shift:\nCICIDS2017 vs CIC-IDS2018",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(
        "results/figures/tsne_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=RANDOM_STATE
    )
    X_umap = reducer.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for ds in colors:
        mask = (y == ds)
        plt.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            label=ds,
            alpha=0.6,
            color=colors[ds]
        )
    plt.title(
        "UMAP Distribution Shift:\nCICIDS2017 vs CIC-IDS2018",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
    plt.ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(
        "results/figures/umap_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    with open("results/metrics/domain_shift_summary.txt", "w") as f:
        f.write("DOMAIN SHIFT ANALYSIS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples used: {len(combined)}\n")
        f.write(f"PCA PC1 variance: {var_exp[0]:.6f}\n")
        f.write(f"PCA PC2 variance: {var_exp[1]:.6f}\n")
    print("\n[INFO] Domain shift analysis completed")
    print("[INFO] Figures saved → results/figures/")
    print("[INFO] Metrics saved → results/metrics/")
