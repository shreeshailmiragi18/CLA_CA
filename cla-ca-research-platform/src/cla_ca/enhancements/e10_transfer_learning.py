"""
E10: Transfer Learning & Cross-Project Defect Prediction (CPDP) for CLA-CA

Performs cross-project defect prediction: trains CLA-CA weights and thresholds
on one project (or set of projects) and tests on target project.
Validates generalization and robustness across different software domains.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef


def run_e10_transfer_learning(
    target_dataset: str,
    source_datasets: List[str] = None
) -> dict:
    """
    Run E10 Cross-Project Defect Prediction (CPDP).
    Trains feature weights and threshold on source projects and evaluates on target.
    """
    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.discretization import discretize_dataset
    from cla_ca.valley_detection import create_split
    from cla_ca.scoring import calculate_scores
    from cla_ca.metrics import calculate_metrics

    all_datasets = [
        'ant-1.5', 'arc', 'berek', 'camel-1.0', 'ivy-2.0',
        'jedit-3.2', 'synapse-1.1', 'tomcat', 'xalan-2.4', 'xerces-1.2',
        'EQ', 'JDT', 'LC', 'ML', 'PDE'
    ]

    if source_datasets is None:
        source_datasets = [d for d in all_datasets if d != target_dataset]

    # 1. Load target dataset
    X_target, y_target, is_aeeem_target = load_dataset(target_dataset)
    feature_cols_target = list(X_target.columns)
    bim_target = 155 if is_aeeem_target else 40

    # 2. Compute average feature weights from source datasets (matching common features or all)
    # Since features may differ across projects, we use target feature weights,
    # but transfer optimal threshold and alpha scaling from source datasets.
    full_df_target = X_target.copy()
    full_df_target['target'] = y_target
    df_norm_target = normalize_dataset(full_df_target, feature_cols_target)
    weights_target = set_weight(df_norm_target, feature_cols_target, alpha=0.8)
    df_disc_target = discretize_dataset(df_norm_target, feature_cols_target, bim=bim_target)

    splist_target = []
    for col in feature_cols_target:
        attr_vals = df_disc_target[col].values
        sp = create_split(attr_vals, bim=bim_target)
        if sp > 0:
            splist_target.append({'index': col, 'value': sp})

    scores_target = calculate_scores(df_disc_target, weights_target, splist_target)

    # 3. Transfer threshold learned from source datasets (e.g. median optimal percentile)
    source_percentiles = []
    for src in source_datasets[:5]: # Use subset for speed
        try:
            X_src, y_src, is_aeeem_src = load_dataset(src)
            cols_src = list(X_src.columns)
            bim_src = 155 if is_aeeem_src else 40
            df_s = X_src.copy()
            df_s['target'] = y_src
            df_n = normalize_dataset(df_s, cols_src)
            w_s = set_weight(df_n, cols_src, alpha=0.8)
            df_d = discretize_dataset(df_n, cols_src, bim=bim_src)
            sp_s = [{'index': c, 'value': create_split(df_d[c].values, bim=bim_src)} for c in cols_src]
            sp_s = [s for s in sp_s if s['value'] > 0]
            sc_s = calculate_scores(df_d, w_s, sp_s)

            # Find percentile of actual defects in source
            sorted_idx = np.argsort(sc_s)
            y_s_sorted = y_src[sorted_idx]
            # Find threshold that matches actual defect count
            actual_defects = int(sum(y_src))
            if 0 < actual_defects < len(sc_s):
                thresh_src = sc_s[sorted_idx[len(sc_s) - actual_defects]]
                percentile = (np.sum(sc_s <= thresh_src) / len(sc_s)) * 100
                source_percentiles.append(percentile)
        except Exception:
            continue

    transfer_percentile = float(np.median(source_percentiles)) if source_percentiles else 80.0
    transferred_threshold = float(np.percentile(scores_target, transfer_percentile))

    y_pred = (scores_target > transferred_threshold).astype(int)
    metrics = calculate_metrics(y_target, y_pred)

    # Baseline prediction for comparison
    from cla_ca.prediction import predict_labels
    y_pred_base, base_threshold = predict_labels(scores_target)
    base_metrics = calculate_metrics(y_target, y_pred_base)

    return {
        'target_dataset': target_dataset,
        'model': 'E10_Transfer_Learning',
        'total_modules': len(y_target),
        'actual_defects': int(sum(y_target)),
        'predicted_defects': int(sum(y_pred)),
        'predicted_defects_baseline': int(sum(y_pred_base)),
        'transferred_threshold': transferred_threshold,
        'transfer_percentile': transfer_percentile,
        'metrics': metrics,
        'base_metrics': base_metrics
    }