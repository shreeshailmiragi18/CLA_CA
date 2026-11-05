"""
E4: Discretization Optimization Enhancement for CLA-CA Algorithm.

Paper baseline: Equal-width discretization with fixed BIM (40 for PROMISE, 155 for AEEEM)

E4 explores:
1. Adaptive BIM selection based on feature distribution
2. Equal-frequency (quantile) discretization
3. Hybrid equal-width + equal-frequency
4. Distribution-aware binning (k-means based)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans


def e4_adaptive_bim_selection(
    df: pd.DataFrame,
    feature: str,
    min_bim: int = 10,
    max_bim: int = 200,
    target_distribution: str = 'uniform'
) -> int:
    """
    E4a: Adaptive BIM selection based on feature characteristics.

    Adjust BIM based on feature variance and skewness.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data
    feature : str
        Feature name
    min_bim : int
        Minimum number of bins
    max_bim : int
        Maximum number of bins
    target_distribution : str
        Target distribution for bins ('uniform', 'skewed')

    Returns
    -------
    bim : int
        Optimal number of bins for this feature
    """
    values = df[feature].values
    n_samples = len(values)

    # Calculate feature statistics
    variance = np.var(values)
    skewness = float(pd.Series(values).skew())

    # Base BIM from paper
    base_bim = min(max_bim, max(min_bim, int(np.sqrt(n_samples))))

    # Adjust based on variance
    if variance > 0.5:
        # High variance feature - more bins needed
        bim_adapted = int(base_bim * 1.2)
    elif variance < 0.1:
        # Low variance feature - fewer bins
        bim_adapted = int(base_bim * 0.7)
    else:
        bim_adapted = base_bim

    # Adjust based on skewness
    if abs(skewness) > 1.0:
        # Highly skewed - reduce bins to avoid empty bins
        bim_adapted = int(bim_adapted * 0.8)

    # Clamp to bounds
    bim_adapted = max(min_bim, min(max_bim, bim_adapted))

    return bim_adapted


def e4_equal_frequency_discretization(
    df: pd.DataFrame,
    features: List[str],
    bim: int = 40
) -> pd.DataFrame:
    """
    E4b: Equal-frequency (quantile) discretization.

    Each bin contains approximately equal number of samples.
    More robust to outliers than equal-width.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized feature matrix
    features : List[str]
        Feature column names
    bim : int
        Number of bins

    Returns
    -------
    df_disc : pd.DataFrame
        Discretized dataframe with 1-based labels
    """
    df_disc = df.copy()

    for col in features:
        values = df[col].values
        # Use pandas qcut for quantile-based discretization
        try:
            # Handle duplicate values
            n_unique = len(np.unique(values))
            if n_unique < bim:
                bim_actual = n_unique
            else:
                bim_actual = bim

            # Quantile discretization
            df_disc[col] = pd.qcut(df[col], q=bim_actual, labels=False, duplicates='drop') + 1
        except:
            # Fallback to equal-width
            df_disc[col] = pd.cut(df[col], bins=bim, labels=False) + 1

    return df_disc


def e4_hybrid_discretization(
    df: pd.DataFrame,
    features: List[str],
    bim_base: int = 40,
    hybrid_ratio: float = 0.7
) -> pd.DataFrame:
    """
    E4c: Hybrid equal-width and equal-frequency discretization.

    Combine both approaches for robustness.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized feature matrix
    features : List[str]
        Feature column names
    bim_base : int
        Base number of bins
    hybrid_ratio : float
        Ratio of equal-width to equal-frequency (0-1)

    Returns
    -------
    df_disc : pd.DataFrame
        Discretized dataframe
    """
    df_disc = df.copy()

    for col in features:
        values = df[col].values

        # Calculate both discretizations
        # Equal-width
        try:
            equal_width = pd.cut(df[col], bins=bim_base, labels=False) + 1
        except:
            equal_width = pd.Series([1] * len(df))

        # Equal-frequency
        try:
            n_unique = len(np.unique(values))
            bim_actual = min(bim_base, n_unique) if n_unique < bim_base else bim_base
            equal_freq = pd.qcut(df[col], q=bim_actual, labels=False, duplicates='drop') + 1
            # Normalize to same range
            if len(equal_freq.unique()) > 0:
                equal_freq = ((equal_freq - 1) / (len(equal_freq.unique()) - 1)) * (bim_base - 1) + 1
        except:
            equal_freq = equal_width

        # Hybrid combination
        hybrid = hybrid_ratio * equal_width + (1 - hybrid_ratio) * equal_freq
        df_disc[col] = hybrid.round().astype(int)
        # Ensure valid range
        df_disc[col] = df_disc[col].clip(1, bim_base)

    return df_disc


def e4_distribution_aware_discretization(
    df: pd.DataFrame,
    features: List[str],
    bim: int = 40
) -> pd.DataFrame:
    """
    E4d: Distribution-aware discretization using k-means clustering.

    Use k-means to find natural clusters in feature values.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized feature matrix
    features : List[str]
        Feature column names
    bim : int
        Number of bins

    Returns
    -------
    df_disc : pd.DataFrame
        Discretized dataframe
    """
    df_disc = df.copy()

    for col in features:
        values = df[col].values.reshape(-1, 1)

        # Check if we have enough unique values
        n_unique = len(np.unique(values))
        if n_unique < 2:
            df_disc[col] = 1
            continue

        # Adjust bins based on data
        n_bins_actual = min(bim, n_unique)

        try:
            # K-means clustering for discretization
            kmeans = KMeans(n_clusters=n_bins_actual, random_state=42, n_init=10)
            labels = kmeans.fit_predict(values)
            # Convert to 1-based labels
            df_disc[col] = labels + 1
        except:
            # Fallback to equal-width
            df_disc[col] = pd.cut(df[col], bins=n_bins_actual, labels=False) + 1

    return df_disc


def run_e4_enhancement(
    dataset_name: str,
    bim: int = None,
    e4_method: str = 'adaptive_bim'
) -> dict:
    """
    Run CLA-CA pipeline with E4 discretization optimization enhancement.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    bim : int, optional
        BIM for discretization
    e4_method : str, optional
        E4 method: 'adaptive_bim', 'equal_frequency', 'hybrid', 'distribution_aware'

    Returns
    -------
    result : dict
        Pipeline result with E4 enhancement
    """
    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.valley_detection import create_split
    from cla_ca.metrics import calculate_metrics
    from cla_ca.prediction import predict_labels
    from cla_ca.scoring import calculate_scores

    # Load dataset
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    # Set base BIM
    if bim is None:
        bim_base = 155 if is_aeeem else 40
    else:
        bim_base = bim

    # Normalize
    feature_cols = list(X_df.columns)
    df_norm = normalize_dataset(X_df, feature_cols)

    # Feature weighting (baseline)
    weights = set_weight(df_norm, feature_cols, alpha=0.8)

    # E4: Discretization optimization
    if e4_method == 'adaptive_bim':
        # Use adaptive BIM per feature
        df_disc_parts = {}
        for col in feature_cols:
            bim_adapted = e4_adaptive_bim_selection(df_norm, col, min_bim=20, max_bim=bim_base)
            df_disc_col = pd.cut(df_norm[col], bins=bim_adapted, labels=False) + 1
            df_disc_parts[col] = df_disc_col
        df_disc = pd.DataFrame(df_disc_parts)

    elif e4_method == 'equal_frequency':
        df_disc = e4_equal_frequency_discretization(df_norm, feature_cols, bim=bim_base)

    elif e4_method == 'hybrid':
        df_disc = e4_hybrid_discretization(df_norm, feature_cols, bim_base=bim_base)

    elif e4_method == 'distribution_aware':
        df_disc = e4_distribution_aware_discretization(df_norm, feature_cols, bim=bim_base)

    else:
        # Baseline equal-width
        from cla_ca.discretization import discretize_dataset
        df_disc = discretize_dataset(df_norm, feature_cols, bim=bim_base)

    # Create split points
    splist = []
    for col in feature_cols:
        # Use baseline BIM for consistency in split detection
        sp = create_split(df_disc[col].values, bim=bim_base)
        if sp > 0:
            splist.append({'index': col, 'value': sp})

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
        'bim': bim_base,
        'e4_method': e4_method,
        'threshold_method': 'baseline',
        'threshold': label_point,
        'metrics': metrics,
        'retained_features': retained_features,
        'total_features': total_features,
        'label_point': label_point,
        'e4_enhanced': True,
        'enhancement_type': 'discretization_optimization',
        'e4_method_used': e4_method
    }

    return result
