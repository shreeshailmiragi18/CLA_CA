"""
E3: Improved Valley Detection Enhancement for CLA-CA Algorithm.

Paper baseline: Algorithm 3 (CreateSplit) finds valleys where adjacent bins have higher frequency
Select split point with maximum (p+q) where p>0 and q>0

E3 explores:
1. Multi-scale valley detection with different smoothing levels
2. Adaptive valley threshold based on distribution shape
3. Valley confidence scoring
4. Multiple split points per feature (hierarchical splits)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def e3_multiscale_valley_detection(
    freq_distribution: np.ndarray,
    bim: int,
    scales: List[float] = [0.5, 1.0, 1.5, 2.0],
    min_prominence: int = 1
) -> List[Dict]:
    """
    E3a: Multi-scale valley detection with smoothing.

    Apply Gaussian smoothing at multiple scales and find consistent valleys.

    Parameters
    ----------
    freq_distribution : np.ndarray
        Frequency counts per bin (length = bim)
    bim : int
        Number of bins
    scales : List[float]
        Gaussian smoothing scales to try
    min_prominence : int
        Minimum prominence for valley detection

    Returns
    -------
    splits : List[Dict]
        List of detected split points with confidence scores
    """
    n_bins = len(freq_distribution)

    # Normalize distribution
    if freq_distribution.sum() > 0:
        freq_norm = freq_distribution / freq_distribution.sum()
    else:
        freq_norm = freq_distribution

    # Find valleys at multiple scales
    all_valleys = []

    for scale in scales:
        # Apply Gaussian smoothing
        if scale > 0.1:
            smoothed = gaussian_filter1d(freq_norm, sigma=scale)
        else:
            smoothed = freq_norm.copy()

        # Find valleys using signal processing
        valleys = signal.find_peaks(-smoothed, prominence=min_prominence / n_bins)
        valley_indices = valleys[0]

        for idx in valley_indices:
            # Confidence based on depth of valley
            valley_depth = smoothed[idx]
            if idx > 0 and idx < len(smoothed) - 1:
                neighbor_avg = (smoothed[idx - 1] + smoothed[idx + 1]) / 2
                confidence = max(0, (neighbor_avg - valley_depth) / (neighbor_avg + 1e-10))
            else:
                confidence = 0.5

            all_valleys.append({
                'index': idx,
                'scale': scale,
                'confidence': confidence,
                'depth': valley_depth
            })

    # Aggregate valleys across scales
    # Group by nearby indices
    if not all_valleys:
        return []

    # Sort by confidence
    all_valleys.sort(key=lambda x: x['confidence'], reverse=True)

    # Select best non-overlapping valleys
    selected_valleys = []
    used_indices = set()
    min_distance = max(2, n_bins // bim * 2)  # Minimum separation

    for valley in all_valleys:
        idx = valley['index']
        # Check if too close to already selected valley
        too_close = False
        for used_idx in used_indices:
            if abs(idx - used_idx) < min_distance:
                too_close = True
                break

        if not too_close:
            selected_valleys.append(valley)
            used_indices.add(idx)

            # Limit to reasonable number of splits
            if len(selected_valleys) >= 2:
                break

    # Format output like baseline
    splits = []
    for valley in selected_valleys:
        splits.append({
            'index': valley['index'],
            'value': valley['index'],
            'confidence': valley['confidence'],
            'scale': valley['scale']
        })

    return splits


def e3_adaptive_valley_threshold(
    freq_distribution: np.ndarray,
    bim: int,
    threshold_factor: float = 0.3
) -> List[Dict]:
    """
    E3b: Adaptive valley threshold based on distribution statistics.

    The prominence threshold adapts to the overall distribution shape.

    Parameters
    ----------
    freq_distribution : np.ndarray
        Frequency counts per bin
    bim : int
        Number of bins
    threshold_factor : float
        Factor for adaptive threshold calculation

    Returns
    -------
    splits : List[Dict]
        List of detected split points
    """
    n_bins = len(freq_distribution)

    # Normalize
    if freq_distribution.sum() > 0:
        freq_norm = freq_distribution / freq_distribution.sum()
    else:
        freq_norm = freq_distribution

    # Calculate adaptive threshold
    mean_freq = np.mean(freq_norm)
    std_freq = np.std(freq_norm)

    # Valleys are points significantly below local average
    adaptive_threshold = mean_freq - threshold_factor * std_freq

    splits = []
    for i in range(1, n_bins - 1):
        # Check if this is a valley
        current = freq_norm[i]
        left = freq_norm[i - 1]
        right = freq_norm[i + 1]

        # Valley condition: current < neighbors AND current < threshold
        if current < left and current < right and current < adaptive_threshold:
            # Calculate valley depth
            valley_depth = min(left, right) - current
            if valley_depth > 0:
                splits.append({
                    'index': i,
                    'value': i,
                    'depth': valley_depth,
                    'confidence': min(1.0, valley_depth / (std_freq + 1e-10))
                })

    # Select best valley (max depth)
    if splits:
        splits.sort(key=lambda x: x['depth'], reverse=True)
        # Keep top 1-2 splits
        best_splits = splits[:2]
        return best_splits

    # Fallback to baseline method
    return []


def e3_ensemble_valley_detection(
    freq_distribution: np.ndarray,
    bim: int,
    methods: List[str] = ['baseline', 'multiscale', 'adaptive']
) -> List[Dict]:
    """
    E3c: Ensemble of valley detection methods.

    Combine multiple valley detection approaches for robustness.

    Parameters
    ----------
    freq_distribution : np.ndarray
        Frequency counts per bin
    bim : int
        Number of bins
    methods : List[str]
        Valley detection methods to ensemble

    Returns
    -------
    splits : List[Dict]
        Consensus split points
    """
    all_candidates = []

    # Baseline method (paper's original)
    if 'baseline' in methods:
        baseline_splits = create_split_baseline(freq_distribution, bim)
        for split in baseline_splits:
            all_candidates.append({
                'index': split['value'],
                'confidence': 1.0,
                'method': 'baseline'
            })

    # Multiscale
    if 'multiscale' in methods:
        multiscale_splits = e3_multiscale_valley_detection(freq_distribution, bim)
        for split in multiscale_splits:
            all_candidates.append({
                'index': split['value'],
                'confidence': split['confidence'],
                'method': 'multiscale'
            })

    # Adaptive threshold
    if 'adaptive' in methods:
        adaptive_splits = e3_adaptive_valley_threshold(freq_distribution, bim)
        for split in adaptive_splits:
            all_candidates.append({
                'index': split['index'],
                'confidence': split['confidence'],
                'method': 'adaptive'
            })

    # Find consensus
    if not all_candidates:
        return []

    # Group by similar indices
    consensus = {}
    for candidate in all_candidates:
        idx = candidate['index']
        # Round to nearest bin
        idx_key = int(idx)

        if idx_key not in consensus:
            consensus[idx_key] = {
                'indices': [],
                'confidence_sum': 0,
                'methods': []
            }

        consensus[idx_key]['indices'].append(idx)
        consensus[idx_key]['confidence_sum'] += candidate['confidence']
        consensus[idx_key]['methods'].append(candidate['method'])

    # Select best consensus splits
    results = []
    for idx_key, data in consensus.items():
        avg_confidence = data['confidence_sum'] / len(data['indices'])
        results.append({
            'index': idx_key,
            'value': idx_key,
            'confidence': avg_confidence,
            'methods_agree': len(data['methods']),
            'methods': data['methods']
        })

    # Sort by confidence and method agreement
    results.sort(key=lambda x: (x['confidence'] * x['methods_agree']), reverse=True)

    # Return top 1-2
    return results[:2]


def create_split_baseline(freq_distribution: np.ndarray, bim: int) -> List[Dict]:
    """
    Baseline valley detection (paper's Algorithm 3).
    """
    n_bins = len(freq_distribution)

    if n_bins < 3:
        return []

    max_valley = -1
    best_idx = -1

    for i in range(1, n_bins - 1):
        if freq_distribution[i] > 0:
            p = freq_distribution[i + 1] - freq_distribution[i]
            q = freq_distribution[i - 1] - freq_distribution[i]

            if p > 0 and q > 0:
                valley_score = p + q
                if valley_score > max_valley:
                    max_valley = valley_score
                    best_idx = i

    if best_idx > 0:
        return [{'index': best_idx, 'value': best_idx, 'confidence': 1.0}]
    return []


def run_e3_enhancement(
    dataset_name: str,
    bim: int = None,
    e3_method: str = 'multiscale'
) -> dict:
    """
    Run CLA-CA pipeline with E3 improved valley detection enhancement.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    bim : int, optional
        BIM for discretization
    e3_method : str, optional
        E3 method: 'multiscale', 'adaptive', 'ensemble'

    Returns
    -------
    result : dict
        Pipeline result with E3 enhancement
    """
    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.discretization import discretize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.metrics import calculate_metrics
    from cla_ca.prediction import predict_labels
    from cla_ca.scoring import calculate_scores

    # Load dataset
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    # Set BIM
    if bim is None:
        bim = 155 if is_aeeem else 40

    # Normalize
    feature_cols = list(X_df.columns)
    df_norm = normalize_dataset(X_df, feature_cols)

    # Feature weighting (baseline)
    weights = set_weight(df_norm, feature_cols, alpha=0.8)

    # Discretization
    df_disc = discretize_dataset(df_norm, feature_cols, bim=bim)

    # E3: Improved valley detection
    splist = []
    for col in feature_cols:
        freq_dist = df_disc[col].value_counts().sort_index()

        # Make sure we have full distribution
        if len(freq_dist) < bim:
            # Pad with zeros
            full_dist = np.zeros(bim)
            for idx, count in freq_dist.items():
                if idx < bim:
                    full_dist[idx] = count
            freq_array = full_dist
        else:
            freq_array = freq_dist.values[:bim]

        if e3_method == 'multiscale':
            splits = e3_multiscale_valley_detection(freq_array, bim)
        elif e3_method == 'adaptive':
            splits = e3_adaptive_valley_threshold(freq_array, bim)
        elif e3_method == 'ensemble':
            splits = e3_ensemble_valley_detection(freq_array, bim)
        else:
            # Baseline fallback
            from cla_ca.valley_detection import create_split
            sp = create_split(df_disc[col].values, bim=bim)
            splits = [{'index': sp, 'value': sp}] if sp > 0 else []

        if splits:
            # Use best split
            best_split = splits[0]
            if best_split['value'] > 0:
                splist.append({
                    'index': col,
                    'value': best_split['value']
                })

    # Calculate scores
    scores = calculate_scores(df_disc, weights, splist)

    # Predict labels
    predictions, label_point = predict_labels(scores)

    # Calculate metrics
    metrics = calculate_metrics(y_true, predictions)

    # Count retained features
    retained_features = sum(1 for s in splist if s['value'] > 0)
    total_features = len(feature_cols)

    result = {
        'dataset': dataset_name,
        'is_aeeem': is_aeeem,
        'bim': bim,
        'e3_method': e3_method,
        'threshold_method': 'baseline',
        'threshold': label_point,
        'metrics': metrics,
        'retained_features': retained_features,
        'total_features': total_features,
        'label_point': label_point,
        'e3_enhanced': True,
        'enhancement_type': 'improved_valley_detection',
        'e3_method_used': e3_method
    }

    return result
