import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from utils.logger import get_logger
logger = get_logger(__name__)
PALETTE    = ['steelblue', 'darkorange', 'seagreen', 'tomato', 'purple']
DPI        = 300
FONT_SIZE  = 11
plt.rcParams.update({
    'font.size':        FONT_SIZE,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.dpi':       DPI,
})
def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray')
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.title(title, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved → {output_path}")
def plot_reconstruction_error(normal_errors, attack_errors,
                               threshold, output_path):
    plt.figure(figsize=(9, 4))
    plt.hist(normal_errors, bins=60, alpha=0.65, color='steelblue',
             label='Normal Traffic', density=True)
    plt.hist(attack_errors, bins=60, alpha=0.65, color='tomato',
             label='Attack Traffic', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold θ = {threshold:.4f}')
    plt.xlabel('Reconstruction Error (MSE)', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title('Autoencoder Reconstruction Error Distribution',
              fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Reconstruction error plot saved → {output_path}")
def plot_roc_curves(models_dict, output_path):
    plt.figure(figsize=(7, 6))
    for i, (name, (fpr, tpr, auc_score)) in enumerate(models_dict.items()):
        plt.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=2,
                 label=f'{name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves — Model Comparison', fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved → {output_path}")
def plot_confidence_distribution(confidence, correct_mask, threshold,
                                  output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(confidence, bins=40, color='steelblue',
                 alpha=0.85, edgecolor='white')
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Routing Threshold ({threshold})')
    axes[0].set_xlabel('Confidence Score', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title('Overall Confidence Distribution', fontweight='bold')
    axes[0].legend()
    axes[1].hist(confidence[correct_mask], bins=40, alpha=0.65,
                 color='steelblue', label='Correct', density=True)
    axes[1].hist(confidence[~correct_mask], bins=40, alpha=0.65,
                 color='tomato', label='Incorrect', density=True)
    axes[1].set_xlabel('Confidence Score', fontweight='bold')
    axes[1].set_ylabel('Density', fontweight='bold')
    axes[1].set_title('Confidence by Prediction Correctness',
                      fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Confidence distribution saved → {output_path}")
def plot_calibration_curve(fraction_correct, mean_confidence,
                            output_path, n_bins=10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5,
                 label='Perfect Calibration')
    axes[0].plot(mean_confidence, fraction_correct,
                 'o-', color='steelblue', linewidth=2,
                 markersize=6, label='Decision Tree')
    axes[0].fill_between(mean_confidence, fraction_correct,
                          mean_confidence, alpha=0.15,
                          color='tomato', label='Calibration Gap')
    axes[0].set_xlabel('Mean Predicted Confidence', fontweight='bold')
    axes[0].set_ylabel('Fraction Correct', fontweight='bold')
    axes[0].set_title('Reliability Diagram', fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].grid(alpha=0.3)
    axes[1].hist(mean_confidence, bins=n_bins, color='steelblue',
                 alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('Confidence Score', fontweight='bold')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('Confidence Histogram', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Calibration curve saved → {output_path}")
def plot_mi_feature_ranking(feature_names, mi_scores, output_path):
    sorted_idx = np.argsort(mi_scores)
    sorted_names  = [feature_names[i] for i in sorted_idx]
    sorted_scores = [mi_scores[i] for i in sorted_idx]
    plt.figure(figsize=(8, 5))
    bars = plt.barh(sorted_names, sorted_scores,
                    color='steelblue', alpha=0.85, edgecolor='white')
    plt.xlabel('Mutual Information Score', fontweight='bold')
    plt.title('Feature Ranking by Mutual Information', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    for bar, score in zip(bars, sorted_scores):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{score:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"MI feature ranking saved → {output_path}")
def plot_generalization_comparison(dataset_labels, results_list,
                                    metrics, output_path):
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    colors = PALETTE[:len(dataset_labels)]
    for idx, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in results_list]
        bars = axes[idx].bar(dataset_labels, values,
                              color=colors, alpha=0.85, edgecolor='white')
        axes[idx].set_ylim(0, 1.05)
        axes[idx].set_title(metric.upper(), fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].tick_params(axis='x', rotation=15)
        axes[idx].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    plt.suptitle('Generalization Performance Across Datasets',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Generalization comparison saved → {output_path}")
def plot_temporal_error_trend(error_series, threshold, alert_idx,
                               output_path, source_ip="192.168.1.x"):
    plt.figure(figsize=(10, 4))
    x = list(range(len(error_series)))
    plt.plot(x, error_series, color='steelblue', linewidth=1.5,
             label='Reconstruction Error per Flow')
    plt.axhline(threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold θ = {threshold:.4f}')
    if alert_idx is not None:
        plt.axvline(alert_idx, color='red', linestyle='-', linewidth=2,
                    label=f'Alert raised (flow #{alert_idx})')
        plt.scatter([alert_idx], [error_series[alert_idx]],
                    color='red', zorder=5, s=80)
    plt.xlabel('Flow Index', fontweight='bold')
    plt.ylabel('Reconstruction Error (MSE)', fontweight='bold')
    plt.title(f'Temporal Error Trend — Source IP: {source_ip}',
              fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Temporal error trend saved → {output_path}")
