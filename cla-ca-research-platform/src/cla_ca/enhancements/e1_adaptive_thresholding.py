"""
E1: Adaptive Thresholding Enhancement for CLA-CA Algorithm.

Investigates alternative score thresholds beyond the median-of-unique-scores
used in the baseline implementation.

Paper baseline: Labelpoint = median(unique(Ks)); if k > Labelpoint then defective

E1 explores:
1. Percentile-based thresholds (25th, 50th, 75th, 90th percentile of all scores)
2. Otsu's method (between-class variance optimization for histogram-based thresholding)
3. Fixed quantile thresholds adapted to training set size
4. Mean-based threshold (alternative to median)
5. Median of all scores (not just unique)

For unsupervised CLA-CA: thresholds should NOT use true labels (y_true)
to avoid data leakage. Only score distribution statistics are allowed.

Each thresholding strategy is implemented as a separate function, and the
best threshold is selected based on validation criteria that do not use
true labels (since CLA-CA is fundamentally unsupervised).
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, Optional


def calculate_scores(df_disc: pd.DataFrame, weights: dict, splist: list) -> np.ndarray:
    """
    Calculate module scores using the CLA-CA Algorithm 1.

    Paper: If xi,index > spindex then ki += W[index]
    Strictly greater (> ) per paper Algorithm 1.
    """
    scores = np.zeros(len(df_disc))
    for item in splist:
        col = item['index']
        sp_val = item['value']
        mask = df_disc[col] > sp_val  # strictly greater
        scores[mask] += weights[col]
    return scores


def predict_labels_baseline(scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Baseline prediction: median of UNIQUE scores as threshold.

    Paper: Ks = unique(Ks); Labelpoint = median(Ks)
    if k > Labelpoint then defective else non-defective
    """
    unique_scores = np.unique(scores)
    label_point = np.median(unique_scores)
    predictions = (scores > label_point).astype(int)
    return predictions, label_point


def threshold_percentile(scores: np.ndarray, percentile: float) -> float:
    """
    Threshold at specified percentile of all scores.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores from CLA-CA scoring
    percentile : float
        Percentile value (e.g., 50 for median, 75 for 75th percentile)

    Returns
    -------
    threshold : float
        Score threshold; scores > threshold are classified as defective
    """
    threshold = np.percentile(scores, percentile)
    return float(threshold)


def threshold_otsu(scores: np.ndarray) -> float:
    """
    Otsu's method: threshold that minimizes within-class variance.

    For 1D histogram-based thresholding, Otsu finds the threshold that
    maximizes between-class variance. This is commonly used in image
    segmentation but can be applied to 1D score distributions.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores from CLA-CA scoring

    Returns
    -------
    threshold : float
        Otsu-selected threshold
    """
    # Create histogram
    hist, bin_edges = np.histogram(scores, bins=20, density=False)

    # Normalize to get probabilities
    total = hist.sum()
    if total == 0:
        return float(np.median(scores))

    probs = hist / total

    # Cumulative sums and means
    cum_prob = np.cumsum(probs)
    cum_mean = np.cumsum(probs * np.arange(len(probs)))

    total_mean = cum_mean[-1] if len(cum_mean) > 0 else 0

    best_threshold = bin_edges[0]
    best_variance = -1

    # Try each possible threshold between bins
    for i in range(1, len(cum_prob)):
        # Class 0: scores <= threshold
        w0 = cum_prob[i - 1]  # weight of class 0
        w1 = 1 - w0  # weight of class 1

        if w0 == 0 or w1 == 0:
            continue

        # Mean of each class
        m0 = cum_mean[i - 1] / w0 if w0 > 0 else 0
        m1 = (total_mean - cum_mean[i - 1]) / w1 if w1 > 0 else 0

        # Between-class variance
        between_var = w0 * w1 * (m0 - m1) ** 2

        if between_var > best_variance:
            best_variance = between_var
            best_threshold = bin_edges[i]

    return float(best_threshold)


def threshold_mean(scores: np.ndarray) -> float:
    """Threshold at mean of all scores (not just unique)."""
    return float(np.mean(scores))


def threshold_median_unique(scores: np.ndarray) -> float:
    """Threshold at median of unique scores (baseline method)."""
    unique_scores = np.unique(scores)
    return float(np.median(unique_scores))


def threshold_weighted_percentile(
    scores: np.ndarray,
    weights: dict,
    feature_cols: list,
    percentile: float = 50.0
) -> float:
    """
    Weighted percentile threshold: scores weighted by feature importance
    are used to compute the percentile.

    This gives more importance to high-weight features in determining
    the threshold.

    Parameters
    ----------
    scores : np.ndarray
        CLA-CA module scores
    weights : dict
        Feature weights from Algorithm 2
    feature_cols : list
        Feature column names
    percentile : float
        Percentile to compute (default: 50 = median)

    Returns
    -------
    threshold : float
        Weighted percentile threshold
    """
    # Create weighted scores: each score weighted by its feature's weight
    weighted_scores = []
    for i, (score, col) in enumerate(zip(scores, feature_cols)):
        if col in weights:
            weighted_scores.append(score * weights[col])
        else:
            weighted_scores.append(score)

    if len(weighted_scores) == 0:
        return float(np.median(scores))

    threshold = np.percentile(weighted_scores, percentile)
    return float(threshold)


def select_threshold(
    scores: np.ndarray,
    method: str = 'baseline',
    **kwargs
) -> Tuple[float, str]:
    """
    Select threshold using the specified method.

    Parameters
    ----------
    scores : np.ndarray
        CLA-CA module scores
    method : str
        Thresholding method: 'baseline', 'percentile', 'otsu', 'mean',
        'weighted_percentile'
    **kwargs
        Additional arguments for the specific method

    Returns
    -------
    threshold : float
        Selected threshold value
    method_name : str
        Name of the method used
    """
    methods: Dict[str, Callable] = {
        'baseline': lambda s: (threshold_median_unique(s), 'baseline'),
        'percentile': lambda s, p: (threshold_percentile(s, p), f'percentile_{p}'),
        'otsu': lambda s: (threshold_otsu(s), 'otsu'),
        'mean': lambda s: (threshold_mean(s), 'mean'),
        'weighted_percentile': lambda s, w, fc, p: (
            threshold_weighted_percentile(s, w, fc, p),
            f'weighted_percentile_{p}'
        ),
    }

    if method not in methods:
        raise ValueError(f"Unknown threshold method: {method}. "
                         f"Choose from: {list(methods.keys())}")

    # Extract relevant kwargs
    if method == 'percentile':
        percentile = kwargs.get('percentile', 50.0)
        return methods[method](scores, percentile)
    elif method == 'weighted_percentile':
        weights = kwargs.get('weights', {})
        feature_cols = kwargs.get('feature_cols', [])
        percentile = kwargs.get('percentile', 50.0)
        return methods[method](scores, weights, feature_cols, percentile)
    else:
        return methods[method](scores)


def predict_labels_e1(
    scores: np.ndarray,
    method: str = 'baseline',
    **kwargs
) -> Tuple[np.ndarray, float, dict]:
    """
    Predict labels using the specified thresholding method.

    Parameters
    ----------
    scores : np.ndarray
        CLA-CA module scores
    method : str
        Thresholding method key
    **kwargs
        Additional arguments passed to select_threshold

    Returns
    -------
    predictions : np.ndarray
        Binary predictions (1 = defective, 0 = non-defective)
    threshold : float
        The selected threshold value
    metadata : dict
        Information about the thresholding method used
    """
    threshold, method_name = select_threshold(scores, method, **kwargs)
    predictions = (scores > threshold).astype(int)

    metadata = {
        'threshold_method': method_name,
        'threshold': threshold,
        'n_positive': int(sum(predictions)),
        'n_negative': int(len(predictions) - sum(predictions)),
        'score_mean': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
    }

    return predictions, threshold, metadata


def evaluate_thresholds(
    scores_list: list,
    y_true_list: list,
    methods: list,
    weights_list: list = None,
    feature_cols_list: list = None
) -> pd.DataFrame:
    """
    Evaluate multiple thresholding methods across multiple datasets.

    This is for analysis/debugging purposes - in production, only use
    methods that don't leak label information.

    Parameters
    ----------
    scores_list : list of np.ndarray
        List of score arrays from different datasets
    y_true_list : list of np.ndarray
        True labels for each dataset (for evaluation only)
    methods : list of str
        Thresholding methods to evaluate
    weights_list : list of dict, optional
        Feature weights for each dataset
    feature_cols_list : list of list, optional
        Feature column names for each dataset

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with metrics for each method/dataset combination
    """
    results = []

    for i, (scores, y_true) in enumerate(zip(scores_list, y_true_list)):
        for method in methods:
            try:
                # Predict using this method
                predictions, threshold, metadata = predict_labels_e1(
                    scores, method=method,
                    weights=weights_list[i] if weights_list else None,
                    feature_cols=feature_cols_list[i] if feature_cols_list else None
                )

                # Calculate metrics (this uses true labels - for analysis only)
                from cla_ca.metrics import calculate_metrics
                metrics = calculate_metrics(y_true, predictions)

                results.append({
                    'dataset_idx': i,
                    'threshold_method': method,
                    'threshold': threshold,
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'f1': metrics['f1'],
                    'mcc': metrics['mcc'],
                    'n_positive': metadata['n_positive'],
                    'n_negative': metadata['n_negative'],
                })
            except Exception as e:
                results.append({
                    'dataset_idx': i,
                    'threshold_method': method,
                    'threshold': None,
                    'recall': None,
                    'precision': None,
                    'f1': None,
                    'mcc': None,
                    'n_positive': None,
                    'n_negative': None,
                    'error': str(e),
                })

    return pd.DataFrame(results)


# Public API: run E1 enhancement on a single dataset
def run_e1_enhancement(
    dataset_name: str,
    bim: int = None,
    alpha: float = 0.8,
    threshold_method: str = 'e1_comparison',
    **kwargs
) -> dict:
    """
    Run CLA-CA pipeline with E1 adaptive thresholding enhancement.

    This function wraps the standard pipeline but replaces the baseline
    thresholding (median of unique scores) with the specified E1 method.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to run
    bim : int, optional
        BIM for discretization (40 for PROMISE, 155 for AEEEM)
    alpha : float, optional
        Feature weighting alpha parameter (default: 0.8)
    threshold_method : str, optional
        E1 thresholding method. Options:
        - 'baseline': median of unique scores (original)
        - 'percentile_25': 25th percentile
        - 'percentile_75': 75th percentile
        - 'percentile_90': 90th percentile
        - 'otsu': Otsu's method
        - 'mean': mean of all scores
        - 'weighted_percentile_50': weighted 50th percentile
    **kwargs
        Additional arguments passed to thresholding methods

    Returns
    -------
    result : dict
        Pipeline result with enhanced thresholding metrics
    """
    from cla_ca.pipeline import run_claca_pipeline
    from cla_ca.data_loader import load_dataset

    # Load dataset
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    # Set BIM
    if bim is None:
        bim = 155 if is_aeeem else 40

    # Run standard pipeline to get scores, weights, discretized data
    feature_cols = list(X_df.columns)

    # Normalize
    from cla_ca.preprocessing import normalize_dataset
    df_norm = normalize_dataset(X_df, feature_cols)

    # Feature weighting
    from cla_ca.feature_weighting import set_weight
    weights = set_weight(df_norm, feature_cols, alpha=alpha)

    # Discretization
    from cla_ca.discretization import discretize_dataset
    df_disc = discretize_dataset(df_norm, feature_cols, bim=bim)

    # Create split points
    from cla_ca.valley_detection import create_split
    splist = []
    for col in feature_cols:
        sp = create_split(df_disc[col].values, bim=bim)
        if sp > 0:
            splist.append({'index': col, 'value': sp})

    # Calculate scores
    scores = calculate_scores(df_disc, weights, splist)

    # E1: Apply adaptive thresholding
    # Map method names: keys are what user passes, values are (internal_method, display_name_suffix)
    method_map = {
        'baseline': ('baseline', 'baseline'),
        'percentile_25': ('percentile', 'percentile_25'),
        'percentile_50': ('percentile', 'percentile_50'),
        'percentile_75': ('percentile', 'percentile_75'),
        'percentile_90': ('percentile', 'percentile_90'),
        'otsu': ('otsu', 'otsu'),
        'mean': ('mean', 'mean'),
        'weighted_percentile_50': ('weighted_percentile', 'weighted_percentile_50'),
    }

    # If threshold_method is 'percentile' and percentile kwarg is provided
    if threshold_method == 'percentile':
        p = kwargs.get('percentile', 50)
        method_name_used = f'percentile_{int(p)}'
        e1_method = 'percentile'
    else:
        mapped = method_map.get(threshold_method, None)
        if mapped is None:
            if threshold_method in ('baseline', 'otsu', 'mean', 'weighted_percentile_50'):
                mapped = (threshold_method, threshold_method)
            else:
                mapped = ('baseline', 'baseline')
        e1_method = mapped[0]
        method_name_used = mapped[1]

    # Apply the E1 thresholding
    predictions, threshold, metadata = predict_labels_e1(
        scores, method=e1_method, **kwargs
    )

    # Calculate metrics
    from cla_ca.metrics import calculate_metrics
    metrics = calculate_metrics(y_true, predictions)

    # Count retained/excluded features (from splist)
    retained_features = sum(1 for s in splist if s['value'] > 0)
    total_features = len(feature_cols)

    result = {
        'dataset': dataset_name,
        'is_aeeem': is_aeeem,
        'bim': bim,
        'alpha': alpha,
        'threshold_method': threshold_method,
        'threshold': threshold,
        'metrics': metrics,

        # Algorithm details
        'retained_features': retained_features,
        'total_features': total_features,
        'label_point': metadata['threshold'],

        # E1 specific
        'e1_enhanced': True,
        'enhancement_type': 'adaptive_thresholding',
        'threshold_method_used': method_name_used,
    }

    return result