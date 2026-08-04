import os
import sys
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.logger import get_logger

logger = get_logger("main")


def stage_preprocess():
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1 — LOCAL DATA PREPROCESSING")
    logger.info("=" * 60)

    from preprocessing.local.pcap_to_csv import run as pcap_to_csv
    from preprocessing.local.clean_flows import clean
    from preprocessing.local.label_flows import label
    from preprocessing.local.extract_all_features import extract
    from preprocessing.local.feature_selection_mi import run as mi_select
    from preprocessing.local.normalize_split import run as normalize

    pcap_to_csv()
    clean()
    label()
    extract()
    mi_select()
    normalize()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 — CICIDS2017 PREPROCESSING")
    logger.info("=" * 60)

    cicids_raw = (
        "data/raw/cicids2017/"
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    )

    if not os.path.exists(cicids_raw):
        logger.warning(
            "CICIDS2017 raw file not found — skipping (download from UNB)"
        )
    else:
        from preprocessing.cicids2017.load_cicids import load
        from preprocessing.cicids2017.align_features import align
        from preprocessing.cicids2017.normalize_cicids import normalize as norm_cicids
        from preprocessing.cicids2017.create_train_split import (
            create_cicids_train_split,
        )

        load()
        align()
        norm_cicids()

        logger.info(
            "Creating CICIDS2017 training splits "
            "(replaces synthetic splits)"
        )
        create_cicids_train_split()

        logger.info(
            "Real training splits ready in data/splits/local/"
        )

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3 — CIC-IDS2018 PREPROCESSING")
    logger.info("=" * 60)

    cic2018_path = (
        "data/raw/cic_ids2018/raw_subset_portscan.csv"
    )

    if not os.path.exists(cic2018_path):
        logger.warning(
            "CIC-IDS2018 raw file not found — skipping (download from UNB)"
        )
    else:
        from preprocessing.cic_ids2018.load_cic2018 import (
            load as load2018,
        )
        from preprocessing.cic_ids2018.sample_balanced_subset import (
            create_subset,
        )
        from preprocessing.cic_ids2018.align_features import (
            align as align2018,
        )
        from preprocessing.cic_ids2018.normalize_cic2018 import (
            normalize as norm2018,
        )

        load2018()
        create_subset()
        align2018()
        norm2018()


def stage_train():
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4 — MODEL TRAINING")
    logger.info("=" * 60)

    cicids_aligned = (
        "data/processed/cicids2017/"
        "cicids_aligned_features.csv"
    )

    if os.path.exists(cicids_aligned):
        logger.info(
            "Using real CICIDS2017 training splits"
        )
    else:
        logger.info(
            "No CICIDS2017 data — using synthetic "
            "training splits"
        )

    from models.classifier.tune_classifier import tune
    from models.classifier.train_classifier import train
    from models.autoencoder.train_autoencoder import (
        train as train_ae,
    )
    from models.autoencoder.threshold_selector import (
        compute_threshold,
    )

    tune()
    train()
    train_ae()
    compute_threshold()


def stage_evaluate():
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5 — INDIVIDUAL MODEL EVALUATION")
    logger.info("=" * 60)

    from models.classifier.evaluate_classifier import (
        run as eval_clf,
    )
    from models.autoencoder.evaluate_autoencoder import (
        run as eval_ae,
    )

    eval_clf()
    eval_ae()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 6 — BASELINES")
    logger.info("=" * 60)

    from baselines.baseline_classifier_only import (
        run as b1,
    )
    from baselines.baseline_autoencoder_only import (
        run as b2,
    )
    from baselines.baseline_fixed_threshold_hybrid import (
        run as b3,
    )

    b1()
    b2()
    b3()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 7 — HYBRID SYSTEM EVALUATION")
    logger.info("=" * 60)

    from hybrid.evaluate_hybrid import (
        run as eval_hybrid,
    )

    eval_hybrid()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 8 — EXPLAINABILITY (SHAP)")
    logger.info("=" * 60)

    try:
        from explainability.shap_explainer import (
            run as shap_run,
        )
        from explainability.plot_shap_summary import (
            run as shap_plot,
        )

        shap_run()
        shap_plot()

    except Exception as e:
        logger.warning(
            f"SHAP failed (install shap library): {e}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 9 — CALIBRATION ANALYSIS")
    logger.info("=" * 60)

    from calibration.calibration_report import (
        run as cal_run,
    )

    cal_run()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 10 — GENERALIZATION SUMMARY")
    logger.info("=" * 60)

    from validation.generalization_summary import (
        run as gen_run,
    )

    gen_run()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 11 — BASELINE COMPARISON TABLE")
    logger.info("=" * 60)

    from baselines.compare_baselines import (
        run as compare_run,
    )

    compare_run()


def stage_analysis():
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 12 — DISTRIBUTION SHIFT ANALYSIS")
    logger.info("=" * 60)

    from domain_shift.pca_visualization import (
        run as pca_run,
    )
    from domain_shift.feature_distribution import (
        run as feat_run,
    )

    pca_run()
    feat_run()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 13 — FEATURE STABILITY (SHAP)")
    logger.info("=" * 60)

    from explainability.shap_analysis import (
        run as shap_run,
    )
    from explainability.stability_analysis import (
        run as stab_run,
    )

    shap_run()
    stab_run()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 14 — CALIBRATION")
    logger.info("=" * 60)

    from calibration.calibration_analysis import (
        run as cal_run,
    )
    from calibration.cross_dataset_calibration import (
        run as cross_cal_run,
    )

    cal_run()
    cross_cal_run()

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 15 — CROSS-DATASET REPORT")
    logger.info("=" * 60)

    from evaluation.cross_dataset_report import (
        run as report_run,
    )

    report_run()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid IDS — Full Pipeline Runner"
    )

    parser.add_argument(
        '--stage',
        default='all',
        choices=[
            'all',
            'preprocess',
            'train',
            'evaluate',
            'analysis'
        ],
        help='Pipeline stage to run '
             '(default: all)'
    )

    args = parser.parse_args()

    t_start = time.time()

    logger.info("=" * 60)
    logger.info("HYBRID IDS — FULL PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Stage: {args.stage}")
    logger.info("Config: config/config.yaml")

    if args.stage in ('all', 'preprocess'):
        stage_preprocess()

    if args.stage in ('all', 'train'):
        stage_train()

    if args.stage in ('all', 'evaluate'):
        stage_evaluate()

    if args.stage in ('all', 'analysis'):
        stage_analysis()

    elapsed = time.time() - t_start

    logger.info(
        f"\nPipeline complete in "
        f"{elapsed:.1f}s"
    )

    logger.info(
        "Results → results/metrics/ "
        "and results/figures/"
    )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()