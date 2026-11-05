"""
E12: Final Optimized CLA-CA with Smart Threshold Selection

Combines:
1. Original CLA-CA pipeline (verified correct implementation)
2. Per-dataset optimal threshold from E1 (Otsu) + E10 (Pareto)
3. Ensemble threshold selection (E5)
4. Defect-rate aware threshold calibration

This is the FINAL optimized version targeting best F1 + fixed overprediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import signal


def threshold_otsu(scores: np.ndarray) -> float:
    """Otsu's method for bimodal threshold selection."""
    from sklearn.preprocessing import StandardScaler
    try:
        hist, bins = np.histogram(scores, bins=256)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        total = hist.sum()
        if total == 0:
            return float(np.median(scores))

        # Otsu's algorithm
        sum_all = (hist * bin_centers).sum()
        weight_b = 0
        sum_b = 0
        max_var = -1
        threshold = 0

        for i in range(len(hist)):
            weight_b += hist[i]
            if weight_b == 0:
                continue
            weight_f = total - weight_b
            if weight_f == 0:
                break
            sum_b += hist[i] * bin_centers[i]
            mean_b = sum_b / weight_b
            mean_f = (sum_all - sum_b) / weight_f
            var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = bin_centers[i]
        return float(threshold)
    except:
        return float(np.median(scores))


def threshold_youden_j(scores: np.ndarray, y_true: np.ndarray = None) -> float:
    """
    Youden's J statistic: maximize (sensitivity + specificity - 1).
    Requires labels, so use cross-validation or fallback to score distribution.
    """
    if y_true is not None:
        # Use actual labels to find optimal threshold
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = y_true[sorted_indices]

        n = len(sorted_labels)
        tps = np.cumsum(sorted_labels)
        fps = np.arange(1, n+1) - tps
        fns = sorted_labels.sum() - tps
        tns = (sorted_labels == 0).sum() - fps

        sensitivities = tps / (tps + fns + 1e-10)
        specificities = tns / (tns + fps + 1e-10)

        youden_j = sensitivities + specificities - 1
        best_idx = np.argmax(youden_j)
        return float(sorted_scores[best_idx])
    else:
        # Fallback: use score distribution
        return float(np.percentile(scores, 75))


def threshold_defect_rate_calibrated(scores: np.ndarray, expected_defect_rate: float = None) -> float:
    """
    Calibrate threshold to match expected defect rate.
    If expected rate known (e.g., from project history), use it.
    Otherwise, estimate from score distribution skewness.
    """
    if expected_defect_rate is not None:
        # Direct percentile
        return float(np.percentile(scores, (1 - expected_defect_rate) * 100))

    # Estimate from score distribution
    # Highly right-skewed = low defect density
    skew = float(pd.Series(scores).skew())

    if skew > 2.0:
        # Very skewed -> very low defect rate ~5-10%
        target_rate = 0.08
    elif skew > 1.0:
        # Moderately skewed -> low defect rate ~10-15%
        target_rate = 0.12
    elif skew > 0.5:
        # Slightly skewed -> moderate defect rate ~15-20%
        target_rate = 0.18
    else:
        # Near symmetric -> higher defect rate ~20-30%
        target_rate = 0.25

    return float(np.percentile(scores, (1 - target_rate) * 100))


def threshold_adaptive_ensemble(scores: np.ndarray) -> Tuple[float, Dict]:
    """
    E12: Ensemble of multiple thresholding methods with agreement weighting.
    """
    methods = {}

    # 1. Baseline (median of unique)
    methods['baseline'] = float(np.median(np.unique(scores)))

    # 2. Otsu
    methods['otsu'] = threshold_otsu(scores)

    # 3. Pareto (80th percentile)
    methods['pareto'] = float(np.percentile(scores, 80))

    # 4. Youden's J (using pseudo-labels from high scores)
    methods['youden'] = threshold_youden_j(scores)

    # 5. Defect-rate calibrated
    methods['calibrated'] = threshold_defect_rate_calibrated(scores)

    # 6. Mean
    methods['mean'] = float(np.mean(scores))

    # 7. 75th percentile
    methods['p75'] = float(np.percentile(scores, 75))

    # Generate predictions from each
    predictions = {}
    for name, thresh in methods.items():
        predictions[name] = (scores > thresh).astype(int)

    # Weighted voting based on expected precision
    # Methods that predict fewer defects get higher weight (precision-oriented)
    pred_counts = {name: np.sum(pred) for name, pred in predictions.items()}
    n_modules = len(scores)
    pred_rates = {name: count / n_modules for name, count in pred_counts.items()}

    # Weight inversely proportional to prediction rate (favor precision)
    weights = {}
    for name, rate in pred_rates.items():
        weights[name] = 1.0 / (rate + 0.01)

    # Normalize
    total_w = sum(weights.values())
    weights = {k: v/total_w for k, v in weights.items()}

    # Weighted ensemble
    weighted_sum = np.zeros(n_modules)
    for name, pred in predictions.items():
        weighted_sum += pred * weights[name]

    ensemble_pred = (weighted_sum > 0.5).astype(int)

    # Find threshold that gives ensemble prediction
    # Sort scores and find threshold matching ensemble count
    sorted_scores = np.sort(scores)
    ensemble_count = int(np.sum(ensemble_pred))
    if ensemble_count > 0 and ensemble_count < n_modules:
        final_threshold = sorted_scores[-(ensemble_count + 1)]
    else:
        final_threshold = methods['baseline']

    return float(final_threshold), {
        'methods': methods,
        'weights': weights,
        'individual_predictions': pred_counts,
        'ensemble_prediction_count': ensemble_count
    }


def run_e12_final_optimized(
    dataset_name: str,
    bim: int = None,
    alpha: float = 0.8
) -> dict:
    """
    Run FINAL optimized CLA-CA pipeline with adaptive ensemble thresholding.
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

    # E12: Adaptive Ensemble Thresholding
    threshold, thresh_meta = threshold_adaptive_ensemble(scores)
    y_pred = (scores > threshold).astype(int)

    # Also get baseline threshold for comparison
    from cla_ca.prediction import predict_labels
    y_pred_base, base_threshold = predict_labels(scores)
    base_metrics = calculate_metrics(y_true, y_pred_base)

    # Metrics
    metrics = calculate_metrics(y_true, y_pred)

    result = {
        'dataset': dataset_name,
        'is_aeeem': is_aeeem,
        'bim': bim,
        'alpha': alpha,
        'total_modules': len(y_true),
        'actual_defects': int(sum(y_true)),
        'predicted_defects': int(sum(y_pred)),
        'predicted_defects_baseline': int(sum(y_pred_base)),
        'threshold': float(threshold),
        'base_threshold': float(base_threshold),
        'metrics': metrics,
        'base_metrics': base_metrics,
        'threshold_metadata': thresh_meta,
        'retained_features': len(retained_features),
        'total_features': len(feature_cols)
    }
    return result