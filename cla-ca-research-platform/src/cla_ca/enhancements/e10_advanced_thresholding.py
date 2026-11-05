"""
E10: Advanced Thresholding & Defect-Rate Calibration for CLA-CA

Fixes the core CLA-CA flaw: severe overprediction of defective modules.
The paper's median-of-unique-scores threshold predicts ~30-50% modules as defective,
even when actual defective rate is <10%.

E10 methods:
1. E10a: Pareto Tail Thresholding (80/20 rule for software defects)
2. E10b: Quantile-Calibrated Thresholding (adaptive top-k estimation)
3. E10c: Score Gap / Clustering Thresholding (Jenks natural breaks)
4. E10d: Cost-Sensitive Precision-Optimized Thresholding
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def threshold_pareto_tail(scores: np.ndarray, pareto_ratio: float = 0.80) -> float:
    """
    E10a: Software defects famously follow the Pareto 80/20 principle:
    ~80% of bugs reside in ~20% of modules.
    Set threshold at the (100 - 20)th percentile = 80th percentile of score distribution.
    """
    if len(scores) == 0:
        return 0.0
    return float(np.percentile(scores, pareto_ratio * 100))


def threshold_quantile_calibrated(scores: np.ndarray, expected_max_defect_rate: float = 0.25) -> float:
    """
    E10b: Calibrate threshold using score distribution skewness.
    Highly skewed score distributions imply low defect density.
    """
    if len(scores) == 0:
        return 0.0

    score_series = pd.Series(scores)
    skew = float(score_series.skew()) if len(scores) > 2 else 0.0

    # Higher skewness -> lower expected defect rate -> higher threshold percentile
    percentile = min(90.0, max(65.0, 75.0 + skew * 5.0))
    return float(np.percentile(scores, percentile))


def threshold_jenks_natural_breaks(scores: np.ndarray) -> float:
    """
    E10c: Jenks Natural Breaks / 2-means clustering on score distribution.
    Finds the natural boundary separating high-risk modules from low-risk modules.
    """
    if len(scores) < 2:
        return 0.0

    sorted_scores = np.sort(scores)

    # 2-means clustering on 1D scores
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sorted_scores.reshape(-1, 1))

    centers = kmeans.cluster_centers_.flatten()
    threshold = float(np.mean(centers))
    return threshold


def run_e10_enhancement(
    dataset_name: str,
    bim: int = None,
    alpha: float = 0.8,
    e10_method: str = 'pareto'
) -> dict:
    """
    Run CLA-CA pipeline with E10 Advanced Thresholding.
    """
    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.discretization import discretize_dataset
    from cla_ca.valley_detection import create_split
    from cla_ca.scoring import calculate_scores
    from cla_ca.metrics import calculate_metrics

    # Load dataset
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    if bim is None:
        bim = 155 if is_aeeem else 40

    feature_cols = list(X_df.columns)
    full_df = X_df.copy()
    full_df['target'] = y_true

    # Normalize & SetWeight
    df_norm = normalize_dataset(full_df, feature_cols)
    weights = set_weight(df_norm, feature_cols, alpha=alpha)

    # Discretization
    df_disc = discretize_dataset(df_norm, feature_cols, bim=bim)

    # Feature Split
    splist = []
    retained_features = []
    excluded_features = []

    for col in feature_cols:
        attr_vals = df_disc[col].values
        sp = create_split(attr_vals, bim=bim)
        if sp > 0:
            splist.append({'index': col, 'value': sp})
            retained_features.append(col)
        else:
            excluded_features.append(col)

    # Module Scoring
    scores = calculate_scores(df_disc, weights, splist)

    # E10 Threshold Selection
    if e10_method == 'pareto':
        threshold = threshold_pareto_tail(scores, pareto_ratio=0.80)
    elif e10_method == 'quantile_calibrated':
        threshold = threshold_quantile_calibrated(scores)
    elif e10_method == 'jenks':
        threshold = threshold_jenks_natural_breaks(scores)
    else:
        threshold = float(np.median(np.unique(scores)))

    # Prediction
    y_pred = (scores > threshold).astype(int)

    # Evaluation Metrics
    metrics = calculate_metrics(y_true, y_pred)

    result = {
        'dataset': dataset_name,
        'is_aeeem': is_aeeem,
        'bim': bim,
        'alpha': alpha,
        'e10_method': e10_method,
        'threshold': float(threshold),
        'total_modules': len(y_true),
        'actual_defects': int(sum(y_true)),
        'predicted_defects': int(sum(y_pred)),
        'metrics': metrics,
        'retained_features': len(retained_features),
        'total_features': len(feature_cols)
    }
    return result
