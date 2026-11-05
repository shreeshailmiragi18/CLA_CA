"""
E5: Ensemble Thresholding Enhancement for CLA-CA Algorithm.

Paper baseline: Single median-of-unique-scores threshold

E5 explores:
1. Ensemble of multiple thresholding methods
2. Weighted voting based on method confidence
3. Adaptive threshold selection per dataset
4. Meta-learning threshold from feature statistics
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def e5_ensemble_thresholding_voting(
    scores: np.ndarray,
    methods: List[str] = ['baseline', 'otsu', 'mean', 'percentile_50'],
    weights: List[float] = None
) -> Tuple[np.ndarray, Dict]:
    """
    E5a: Ensemble voting with multiple thresholding methods.

    Combine predictions from multiple thresholding methods using voting.

    Parameters
    ----------
    scores : np.ndarray
        CLA-CA module scores
    methods : List[str]
        Thresholding methods to ensemble
    weights : List[float], optional
        Weights for each method (default: equal weights)

    Returns
    -------
    predictions : np.ndarray
        Ensemble predictions
    metadata : Dict
        Ensemble metadata
    """
    from cla_ca.enhancements.e1_adaptive_thresholding import (
        predict_labels_baseline, threshold_otsu, threshold_mean, threshold_percentile
    )

    predictions_list = []
    thresholds = {}

    # Generate predictions from each method
    for method in methods:
        if method == 'baseline':
            pred, thresh = predict_labels_baseline(scores)
        elif method == 'otsu':
            thresh = threshold_otsu(scores)
            pred = (scores > thresh).astype(int)
        elif method == 'mean':
            thresh = float(np.mean(scores))
            pred = (scores > thresh).astype(int)
        elif method.startswith('percentile_'):
            percentile = int(method.split('_')[1])
            thresh = threshold_percentile(scores, percentile)
            pred = (scores > thresh).astype(int)
        else:
            continue

        predictions_list.append(pred)
        thresholds[method] = thresh

    # Voting
    if weights is None:
        weights = [1.0] * len(predictions_list)

    # Weighted voting
    predictions_array = np.vstack(predictions_list)
    weights_array = np.array(weights)

    # Weighted sum
    weighted_sum = np.sum(predictions_array * weights_array[:, np.newaxis], axis=0)
    threshold_vote = np.sum(weights_array) / 2

    ensemble_predictions = (weighted_sum > threshold_vote).astype(int)

    metadata = {
        'methods_used': methods,
        'thresholds': thresholds,
        'individual_predictions': predictions_list,
        'ensemble_size': len(methods)
    }

    return ensemble_predictions, metadata


def e5_confidence_weighted_ensemble(
    scores: np.ndarray,
    threshold_methods: List[str] = ['baseline', 'otsu', 'mean'],
    confidence_metrics: List[str] = ['variance', 'skewness']
) -> Tuple[np.ndarray, Dict]:
    """
    E5b: Confidence-weighted ensemble based on score distribution.

    Weight each method by how well it matches the score distribution characteristics.

    Parameters
    ----------
    scores : np.ndarray
        CLA-CA module scores
    threshold_methods : List[str]
        Methods to ensemble
    confidence_metrics : List[str]
        Metrics to assess method confidence

    Returns
    -------
    predictions : np.ndarray
        Ensemble predictions
    metadata : Dict
        Ensemble metadata
    """
    from cla_ca.enhancements.e1_adaptive_thresholding import (
        predict_labels_baseline, threshold_otsu, threshold_mean
    )

    # Calculate score distribution characteristics
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    score_skew = float(pd.Series(scores).skew()) if len(scores) > 0 else 0
    score_kurt = float(pd.Series(scores).kurtosis()) if len(scores) > 0 else 0

    # Calculate method confidences
    method_confidences = {}
    method_predictions = {}
    method_thresholds = {}

    for method in threshold_methods:
        if method == 'baseline':
            pred, thresh = predict_labels_baseline(scores)
            # Confidence: baseline works well for normal distributions
            confidence = 1.0 / (1.0 + abs(score_skew) * 0.1)
        elif method == 'otsu':
            thresh = threshold_otsu(scores)
            pred = (scores > thresh).astype(int)
            # Confidence: Otsu works well for bimodal distributions
            confidence = 1.0 / (1.0 + abs(score_kurt - 3) * 0.1)
        elif method == 'mean':
            thresh = float(np.mean(scores))
            pred = (scores > thresh).astype(int)
            # Confidence: mean works well for symmetric distributions
            confidence = 1.0 / (1.0 + abs(score_skew) * 0.2)
        else:
            continue

        method_confidences[method] = confidence
        method_predictions[method] = pred
        method_thresholds[method] = thresh

    # Normalize confidences to weights
    conf_values = list(method_confidences.values())
    total_conf = sum(conf_values)
    if total_conf > 0:
        weights = {m: c/total_conf for m, c in method_confidences.items()}
    else:
        weights = {m: 1.0/len(method_confidences) for m in method_confidences}

    # Weighted voting
    predictions_array = np.vstack([method_predictions[m] for m in method_confidences])
    weights_array = np.array([weights[m] for m in method_confidences])

    weighted_sum = np.sum(predictions_array * weights_array[:, np.newaxis], axis=0)
    threshold_vote = 0.5

    ensemble_predictions = (weighted_sum > threshold_vote).astype(int)

    metadata = {
        'methods_used': list(method_confidences.keys()),
        'method_thresholds': method_thresholds,
        'method_confidences': method_confidences,
        'method_weights': weights,
        'score_stats': {
            'mean': score_mean,
            'std': score_std,
            'skewness': score_skew,
            'kurtosis': score_kurt
        }
    }

    return ensemble_predictions, metadata


def e5_meta_learning_threshold(
    scores: np.ndarray,
    feature_stats: Dict[str, float],
    n_iterations: int = 10
) -> Tuple[float, Dict]:
    """
    E5c: Meta-learning threshold based on feature statistics and score patterns.

    Train a simple model to predict optimal threshold from score distribution.

    Parameters
    ----------
    scores : np.ndarray
        CLA-CA module scores
    feature_stats : Dict[str, float]
        Statistics about features (variance, correlation, etc.)
    n_iterations : int
        Number of iterations for meta-learning

    Returns
    -------
    threshold : float
        Meta-learned threshold
    metadata : Dict
        Meta-learning metadata
    """
    # Extract score distribution features
    score_features = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'skewness': float(pd.Series(scores).skew()) if len(scores) > 0 else 0,
        'kurtosis': float(pd.Series(scores).kurtosis()) if len(scores) > 0 else 0,
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75),
        'n_unique': len(np.unique(scores))
    }

    # Combine with feature stats
    all_features = {**score_features, **feature_stats}

    # Simple heuristic-based threshold selection
    # This is a simplified version - in practice would use historical data
    base_threshold = np.median(np.unique(scores))

    # Adjust based on distribution shape
    skew = all_features['skewness']
    if abs(skew) > 1.0:
        # Highly skewed - use percentile-based adjustment
        if skew > 0:
            # Right skewed - threshold higher
            threshold = np.percentile(scores, 60)
        else:
            # Left skewed - threshold lower
            threshold = np.percentile(scores, 40)
    else:
        # Near symmetric - use median
        threshold = base_threshold

    # Adjust based on variance
    std_ratio = all_features['std'] / (all_features['mean'] + 1e-10)
    if std_ratio > 1.0:
        # High variance - be more conservative
        threshold = threshold * 1.1
    elif std_ratio < 0.3:
        # Low variance - be more aggressive
        threshold = threshold * 0.9

    metadata = {
        'score_features': score_features,
        'feature_stats': feature_stats,
        'base_threshold': base_threshold,
        'adjusted_threshold': threshold,
        'adjustment_factors': {
            'skewness': skew,
            'std_ratio': std_ratio
        }
    }

    return float(threshold), metadata


def run_e5_enhancement(
    dataset_name: str,
    bim: int = None,
    alpha: float = 0.8,
    e5_method: str = 'ensemble_voting'
) -> dict:
    """
    Run CLA-CA pipeline with E5 ensemble thresholding enhancement.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    bim : int, optional
        BIM for discretization
    alpha : float, optional
        Feature weighting alpha parameter
    e5_method : str, optional
        E5 method: 'ensemble_voting', 'confidence_weighted', 'meta_learning'

    Returns
    -------
    result : dict
        Pipeline result with E5 enhancement
    """
    import pandas as pd
    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.discretization import discretize_dataset
    from cla_ca.valley_detection import create_split
    from cla_ca.metrics import calculate_metrics
    from cla_ca.scoring import calculate_scores

    # Load dataset
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    # Set BIM
    if bim is None:
        bim = 155 if is_aeeem else 40

    # Normalize
    feature_cols = list(X_df.columns)
    df_norm = normalize_dataset(X_df, feature_cols)

    # Feature weighting
    weights = set_weight(df_norm, feature_cols, alpha=alpha)

    # Discretization
    df_disc = discretize_dataset(df_norm, feature_cols, bim=bim)

    # Create split points
    splist = []
    for col in feature_cols:
        sp = create_split(df_disc[col].values, bim=bim)
        if sp > 0:
            splist.append({'index': col, 'value': sp})

    # Calculate scores
    scores = calculate_scores(df_disc, weights, splist)

    # E5: Ensemble thresholding
    if e5_method == 'ensemble_voting':
        predictions, metadata = e5_ensemble_thresholding_voting(
            scores,
            methods=['baseline', 'otsu', 'mean', 'percentile_50']
        )
        threshold = metadata['thresholds'].get('baseline', np.median(scores))

    elif e5_method == 'confidence_weighted':
        predictions, metadata = e5_confidence_weighted_ensemble(scores)
        # Use weighted average of thresholds
        thresholds = list(metadata['method_thresholds'].values())
        threshold = np.mean(thresholds)

    elif e5_method == 'meta_learning':
        # Collect feature statistics
        feature_stats = {}
        for col in feature_cols:
            feature_stats[f'{col}_var'] = float(df_norm[col].var())
            feature_stats[f'{col}_mean'] = float(df_norm[col].mean())

        threshold, metadata = e5_meta_learning_threshold(scores, feature_stats)
        predictions = (scores > threshold).astype(int)

    else:
        # Fallback to baseline
        from cla_ca.prediction import predict_labels
        predictions, threshold = predict_labels(scores)
        metadata = {}

    # Calculate metrics
    metrics = calculate_metrics(y_true, predictions)

    # Count retained features
    retained_features = sum(1 for s in splist if s['value'] > 0)
    total_features = len(feature_cols)

    result = {
        'dataset': dataset_name,
        'is_aeeem': is_aeeem,
        'bim': bim,
        'alpha': alpha,
        'e5_method': e5_method,
        'threshold': threshold,
        'metrics': metrics,
        'retained_features': retained_features,
        'total_features': total_features,
        'label_point': threshold,
        'e5_enhanced': True,
        'enhancement_type': 'ensemble_thresholding',
        'e5_method_used': e5_method,
        'e5_metadata': metadata
    }

    return result
