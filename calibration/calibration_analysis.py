   
                                   

                                   
                                                                 
                                                                  

         
                                                  
                                         
   

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')

MODEL_PATH  = "models/classifier/saved/decision_tree_model.pkl"

TRAIN_X     = "data/splits/cicids2017/X_train.csv"
TRAIN_Y     = "data/splits/cicids2017/y_train.csv"

CICIDS_X    = "data/splits/cicids2017/X_test_cicids.csv"
CICIDS_Y    = "data/splits/cicids2017/y_test_cicids.csv"
IDS2018_X   = "data/splits/cic_ids2018/X_test_cic2018.csv"
IDS2018_Y   = "data/splits/cic_ids2018/y_test_cic2018.csv"


FIG_OUT     = "results/figures/figure4_calibration_analysis.png"
METRICS_OUT = "results/metrics/calibration_report.json"

N_BINS = 10


def compute_ece(y_true, y_prob, n_bins=10):
       
                               
                                                  

            
                               
                                                      
       
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    bin_data = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        w    = mask.sum() / n
        ece += w * abs(acc - conf)
        bin_data.append({
            "bin":        f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            "n":          int(mask.sum()),
            "accuracy":   round(acc, 4),
            "confidence": round(conf, 4),
            "gap":        round(abs(acc - conf), 4),
            "weight":     round(float(w), 4),
        })

    return float(ece), bin_data


def get_binary_proba(clf, X):
       
                                                                       
                                                                       
                                                                        
       
    proba = clf.predict_proba(X)

    if proba.shape[1] == 2:
                                                   
        return proba[:, 1]
    else:
                                                              
                                              
        return proba[:, 1:].sum(axis=1)


from sklearn.metrics import roc_auc_score, roc_curve

def compute_auc(y_true, y_score):
                                                                    
    try:
        if len(np.unique(y_true)) == 2:
            auc_score = roc_auc_score(y_true, y_score)
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        else:
            auc_score = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
            fpr_arr, tpr_arr, _ = np.array([0,1]), np.array([0,1]), None
        return float(auc_score), fpr_arr, tpr_arr
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        return 0.5, np.array([0, 1]), np.array([0, 1])



def load_dataset(x_path, y_path, name, n_features=10):
                                                                       
    if os.path.exists(x_path) and os.path.exists(y_path):
        X_df = pd.read_csv(x_path)
        meta = ['label', 'label_str']
        feat = [c for c in X_df.columns if c not in meta]
        X    = X_df[feat].values
        y    = pd.read_csv(y_path).values.ravel().astype(int)
                              
        y = (y != 0).astype(int)
        logger.info(f"Loaded {name}: {len(X)} samples")
        return X, y
    else:
        logger.warning(f"{name} not found — generating synthetic demo data")
        np.random.seed(abs(hash(name)) % 999)
        n = 800
        X = np.random.rand(n, n_features)
        y = np.random.randint(0, 2, n)
        return X, y


def run():
                                                               
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Run main.py first to train the classifier")
        return None

    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    logger.info(f"Classifier loaded: depth={clf.get_depth()}, "
                f"classes={clf.classes_}")

    from sklearn.calibration import CalibratedClassifierCV

                                       
    X_train = pd.read_csv(TRAIN_X).values
    y_train = pd.read_csv(TRAIN_Y).values.ravel().astype(int)
    y_train = (y_train != 0).astype(int)                 

    calibrated_clf = CalibratedClassifierCV(
        base_estimator=clf,
        method="isotonic",                
        cv=5
    )
    calibrated_clf.fit(X_train, y_train)
    clf = calibrated_clf

                                                               
    datasets = [
        ("CICIDS2017\n(In-Domain)",   CICIDS_X,  CICIDS_Y,  '#2196F3'),
        ("CIC-IDS2018\n(Cross-Domain)", IDS2018_X, IDS2018_Y, '#FF9800'),
    ]

                                                               
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                             hspace=0.45, wspace=0.35)

    all_metrics = {}
    roc_data    = {}

    for col, (name, x_path, y_path, color) in enumerate(datasets):
        X, y = load_dataset(x_path, y_path, name)

                                     
        y_prob = get_binary_proba(clf, X)
        y_pred = clf.predict(X)
        y_pred_binary = (y_pred != 0).astype(int)
        correct = (y_pred_binary == y)

             
        ece, bin_data = compute_ece(y, y_prob, n_bins=N_BINS)

             
        auc_score, fpr_arr, tpr_arr = compute_auc(y, y_prob)
        roc_data[name.replace('\n', ' ')] = {
            'fpr': fpr_arr, 'tpr': tpr_arr, 'auc': auc_score}

                           
        frac_pos, mean_pred = calibration_curve(
            y, y_prob, n_bins=10, strategy='uniform'
        )

                          
        mean_conf_correct   = float(y_prob[correct].mean())   if correct.any()  else 0.0
        mean_conf_incorrect = float(y_prob[~correct].mean())  if (~correct).any() else 0.0
        pct_high_conf       = float((y_prob >= 0.85).mean())

        all_metrics[name.replace('\n', ' ')] = {
            "ece":                  round(ece, 4),
            "auc":                  round(auc_score, 4),
            "mean_conf_correct":    round(mean_conf_correct, 4),
            "mean_conf_incorrect":  round(mean_conf_incorrect, 4),
            "pct_high_confidence":  round(pct_high_conf, 4),
            "n_samples":            int(len(y)),
            "bin_details":          bin_data,
        }

        logger.info(f"\n{name.strip()}:")
        logger.info(f"  ECE:  {ece:.4f}")
        logger.info(f"  AUC:  {auc_score:.4f}")
        logger.info(f"  Mean conf (correct):   {mean_conf_correct:.4f}")
        logger.info(f"  Mean conf (incorrect): {mean_conf_incorrect:.4f}")

                                                               
        ax = fig.add_subplot(gs[0, col])
        ax.plot([0,1],[0,1], 'k--', alpha=0.5, linewidth=1.5,
                label='Perfect calibration')
        ax.plot(mean_pred, frac_pos, 'o-', color=color, linewidth=2.5,
                markersize=7, label=f'ECE = {ece:.4f}')
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        alpha=0.18, color=color, label='Calibration gap')
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontweight='bold')
        ax.set_title(f'Reliability Diagram\n{name}',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim([0,1]); ax.set_ylim([0,1])
        ax.grid(alpha=0.3)

                                                               
        ax2 = fig.add_subplot(gs[1, col])
        ax2.hist(y_prob[y==0], bins=40, density=True, alpha=0.65,
                 color='#4CAF50', label='Normal (true)')
        ax2.hist(y_prob[y==1], bins=40, density=True, alpha=0.65,
                 color='#F44336', label='Attack (true)')
        ax2.axvline(0.5, color='black', linestyle='--',
                    linewidth=1.5, alpha=0.7, label='Decision boundary')
        ax2.set_xlabel('Predicted P(Attack)', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title(f'Confidence Distribution\n{name}',
                      fontweight='bold', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

                                                               
    ax_roc = fig.add_subplot(gs[0, 2])
    roc_colors = ['#2196F3', '#FF9800']
    for (ds_name, rd), rc in zip(roc_data.items(), roc_colors):
        ax_roc.plot(rd['fpr'], rd['tpr'], color=rc, linewidth=2,
                    label=f"{ds_name.split('(')[0].strip()} "
                          f"(AUC={rd['auc']:.3f})")
    ax_roc.plot([0,1],[0,1], 'k--', alpha=0.4, linewidth=1)
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
    ax_roc.set_title('ROC Curves — Dataset Comparison',
                     fontweight='bold', fontsize=10)
    ax_roc.legend(fontsize=8, loc='lower right')
    ax_roc.grid(alpha=0.3)

                                                               
    ax_ece = fig.add_subplot(gs[1, 2])
    ds_names_short = [k.split('(')[0].strip() for k in all_metrics.keys()]
    ece_vals       = [v['ece'] for v in all_metrics.values()]
    bars = ax_ece.bar(ds_names_short, ece_vals,
                      color=['#2196F3', '#FF9800'],
                      alpha=0.85, edgecolor='white', width=0.5)
    ax_ece.set_ylabel('Expected Calibration Error', fontweight='bold')
    ax_ece.set_title('ECE Comparison\n(lower = better calibrated)',
                     fontweight='bold', fontsize=10)
    ax_ece.axhline(0.05, color='green',  linestyle='--', alpha=0.7,
                   label='Well calibrated (<0.05)')
    ax_ece.axhline(0.10, color='orange', linestyle='--', alpha=0.7,
                   label='Acceptable (<0.10)')
    ax_ece.axhline(0.15, color='red',    linestyle='--', alpha=0.7,
                   label='Poor (>0.15)')
    ax_ece.legend(fontsize=7)
    ax_ece.grid(axis='y', alpha=0.3)
    ax_ece.set_ylim(0, max(ece_vals) * 1.5 if max(ece_vals) > 0 else 0.2)
    for bar, val in zip(bars, ece_vals):
        ax_ece.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

                                                               
    ax_gap = fig.add_subplot(gs[2, :])
    colors_bar = ['#2196F3', '#FF9800']
    labels_bar = [k.split('(')[0].strip() for k in all_metrics.keys()]
    width = 0.35

    for idx, (ds_key, ds_meta, ds_col, ds_lbl) in enumerate(
            zip(all_metrics.keys(),
                all_metrics.values(),
                colors_bar, labels_bar)):
        bins_x    = [b['confidence'] for b in ds_meta['bin_details']]
        gaps      = [b['gap']        for b in ds_meta['bin_details']]
        offsets   = [x + (idx - 0.5) * width for x in bins_x]
        ax_gap.bar(offsets, gaps, width * 0.9, color=ds_col,
                   alpha=0.8, label=ds_lbl, edgecolor='white')

    ax_gap.set_xlabel('Mean Predicted Probability (bin center)',
                      fontweight='bold')
    ax_gap.set_ylabel('|Accuracy − Confidence|', fontweight='bold')
    ax_gap.set_title('Calibration Gap per Confidence Bin\n'
                     '(taller bar = worse calibration in that region)',
                     fontweight='bold')
    ax_gap.legend(fontsize=9)
    ax_gap.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Calibration Analysis: Reliability, Confidence Distributions, and ECE',
        fontsize=14, fontweight='bold', y=1.01)

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"\nCalibration figure saved → {FIG_OUT}")

                               
    print("\n" + "="*65)
    print("CALIBRATION REPORT — FOR PAPER SECTION 6")
    print("="*65)
    for ds, m in all_metrics.items():
        print(f"\n{ds}:")
        print(f"  ECE:                     {m['ece']:.4f}")
        print(f"  AUC:                     {m['auc']:.4f}")
        print(f"  Mean conf (correct):     {m['mean_conf_correct']:.4f}")
        print(f"  Mean conf (incorrect):   {m['mean_conf_incorrect']:.4f}")
        print(f"  % high-conf predictions: {m['pct_high_confidence']*100:.1f}%")
    print("="*65)

                  
    serializable = {k: {kk: vv for kk, vv in v.items() if kk != 'bin_details'}
                    for k, v in all_metrics.items()}
    for k in all_metrics:
        serializable[k]['bin_details'] = all_metrics[k]['bin_details']

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Calibration metrics saved → {METRICS_OUT}")

    return all_metrics


if __name__ == "__main__":
    run()