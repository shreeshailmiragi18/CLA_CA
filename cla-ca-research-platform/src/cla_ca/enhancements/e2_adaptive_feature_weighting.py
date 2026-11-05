"""
E2: Adaptive Feature Weighting Enhancement for CLA-CA Algorithm.

Paper baseline: Algorithm 2 uses variance-based weighting with fixed alpha=0.8
Weight = e (if var <= threshold) or e + alpha * var/maxvar

E2 explores:
1. Multi-level alpha adaptation based on feature correlation
2. Dynamic ratio threshold adjustment
3. Weighted combination of variance and information gain
4. Feature stability weighting across bootstraps

For unsupervised CLA-CA: we cannot use labels, but can use feature statistics and stability metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.preprocessing import MinMaxScaler


def calculate_feature_variance(df: pd.DataFrame, features: list) -> Dict[str, float]:
    """Calculate variance for each feature."""
    variances = {}
    for col in features:
        variances[col] = float(df[col].var())
    return variances


def calculate_feature_correlation(df: pd.DataFrame, features: list) -> np.ndarray:
    """Calculate correlation matrix for features."""
    corr_matrix = df[features].corr().values
    return corr_matrix


def e2_adaptive_alpha_weighting(
    df: pd.DataFrame,
    features: list,
    alpha_base: float = 0.8,
    correlation_threshold: float = 0.7
) -> Dict[str, float]:
    """
    E2a: Alpha adaptation based on feature correlation.

    Features that are highly correlated should have reduced individual weight
    to avoid double counting the same information.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized feature matrix
    features : list
        List of feature column names
    alpha_base : float
        Base alpha value from paper
    correlation_threshold : float
        Threshold for considering features highly correlated

    Returns
    -------
    weights : dict
        Feature weights adjusted for correlation
    """
    # Calculate variances
    variances = calculate_feature_variance(df, features)

    # Calculate correlation matrix
    corr_matrix = calculate_feature_correlation(df, features)

    # Compute correlation-based penalty for each feature
    weights = {}
    n_features = len(features)

    for i, col in enumerate(features):
        var = variances[col]

        # Find highly correlated features
        correlated_count = 0
        for j in range(n_features):
            if i != j:
                corr_val = abs(corr_matrix[i, j])
                if corr_val > correlation_threshold:
                    correlated_count += 1

        # Reduce alpha for highly correlated features
        correlation_penalty = 1.0 / (1.0 + (correlated_count * 0.1))
        alpha_adapted = alpha_base * correlation_penalty

        # Apply paper's formula with adapted alpha
        var_values = list(variances.values())
        max_var = max(var_values) if var_values else 1.0
        ratio_threshold = 1.5  # Paper's ratio threshold

        if var <= ratio_threshold * (max_var / len(features)):
            weight = 1.0
        else:
            weight = 1.0 + alpha_adapted * (var / max_var)

        weights[col] = weight

    return weights


def e2_stability_weighting(
    df: pd.DataFrame,
    features: list,
    n_bootstrap: int = 20,
    alpha_base: float = 0.8
) -> Dict[str, float]:
    """
    E2b: Stability weighting using bootstrap sampling.

    Features with stable variance across bootstrap samples are more reliable.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized feature matrix
    features : list
        List of feature column names
    n_bootstrap : int
        Number of bootstrap samples
    alpha_base : float
        Base alpha value

    Returns
    -------
    weights : dict
        Feature weights adjusted for stability
    """
    n_samples = len(df)
    variance_stability = {}

    # Calculate variance stability via bootstrapping
    for col in features:
        variances = []
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sample = df.iloc[indices][col]
            variances.append(sample.var())

        # Stability = 1 / coefficient of variation
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        if mean_var > 0:
            cv = std_var / mean_var
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 1.0

        variance_stability[col] = stability

    # Calculate base weights using paper formula
    variances = calculate_feature_variance(df, features)
    var_values = list(variances.values())
    max_var = max(var_values) if var_values else 1.0

    weights = {}
    for col in features:
        var = variances[col]
        stability = variance_stability[col]

        # Adapt alpha based on stability
        alpha_adapted = alpha_base * stability

        # Apply paper's formula
        ratio_threshold = 1.5
        if var <= ratio_threshold * (max_var / len(features)):
            weight = 1.0
        else:
            weight = 1.0 + alpha_adapted * (var / max_var)

        weights[col] = weight

    return weights


def e2_information_gain_weighting(
    df: pd.DataFrame,
    features: list,
    alpha_base: float = 0.8
) -> Dict[str, float]:
    """
    E2c: Combine variance with information gain (entropy-based).

    For unsupervised case, use feature distribution entropy as proxy for information.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized feature matrix
    features : list
        List of feature column names
    alpha_base : float
        Base alpha value

    Returns
    -------
    weights : dict
        Feature weights combining variance and entropy
    """
    variances = calculate_feature_variance(df, features)
    var_values = list(variances.values())
    max_var = max(var_values) if var_values else 1.0

    weights = {}
    for col in features:
        var = variances[col]

        # Calculate normalized entropy (information content)
        # Use histogram bins to estimate distribution entropy
        values = df[col].values
        hist, _ = np.histogram(values, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        if len(hist) > 0:
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            # Normalize entropy to [0, 1]
            max_entropy = np.log2(10)
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.5

        # Combine variance and entropy
        var_score = var / max_var
        info_score = normalized_entropy

        # Adaptive alpha based on information content
        alpha_adapted = alpha_base * (0.7 + 0.3 * info_score)

        # Apply paper's formula with enhanced weighting
        ratio_threshold = 1.5
        if var <= ratio_threshold * (max_var / len(features)):
            weight = 1.0 + 0.1 * info_score
        else:
            weight = 1.0 + alpha_adapted * var_score * (1.0 + 0.2 * info_score)

        weights[col] = float(weight)

    return weights


def run_e2_enhancement(
    dataset_name: str,
    bim: int = None,
    alpha: float = 0.8,
    e2_method: str = 'correlation_adapted'
) -> dict:
    """
    Run CLA-CA pipeline with E2 adaptive feature weighting enhancement.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    bim : int, optional
        BIM for discretization
    alpha : float, optional
        Feature weighting alpha parameter
    e2_method : str, optional
        E2 method: 'correlation_adapted', 'stability', 'information_gain'

    Returns
    -------
    result : dict
        Pipeline result with E2 enhancement
    """
    from cla_ca.pipeline import run_claca_pipeline
    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.discretization import discretize_dataset
    from cla_ca.valley_detection import create_split
    from cla_ca.metrics import calculate_metrics

    # Load dataset
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    # Set BIM
    if bim is None:
        bim = 155 if is_aeeem else 40

    # Normalize
    feature_cols = list(X_df.columns)
    df_norm = normalize_dataset(X_df, feature_cols)

    # E2: Apply adaptive feature weighting
    if e2_method == 'correlation_adapted':
        # Import from this module
        weights = e2_adaptive_alpha_weighting(df_norm, feature_cols, alpha_base=alpha)
    elif e2_method == 'stability':
        weights = e2_stability_weighting(df_norm, feature_cols, alpha_base=alpha)
    elif e2_method == 'information_gain':
        weights = e2_information_gain_weighting(df_norm, feature_cols, alpha_base=alpha)
    else:
        # Fallback to baseline
        from cla_ca.feature_weighting import set_weight
        weights = set_weight(df_norm, feature_cols, alpha=alpha)

    # Discretization
    df_disc = discretize_dataset(df_norm, feature_cols, bim=bim)

    # Create split points
    from cla_ca.scoring import calculate_scores
    splist = []
    for col in feature_cols:
        sp = create_split(df_disc[col].values, bim=bim)
        if sp > 0:
            splist.append({'index': col, 'value': sp})

    # Calculate scores
    scores = calculate_scores(df_disc, weights, splist)

    # Predict labels (using baseline thresholding)
    from cla_ca.prediction import predict_labels
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
        'alpha': alpha,
        'e2_method': e2_method,
        'threshold_method': 'baseline',
        'threshold': label_point,
        'metrics': metrics,
        'retained_features': retained_features,
        'total_features': total_features,
        'label_point': label_point,
        'e2_enhanced': True,
        'enhancement_type': 'adaptive_feature_weighting',
        'e2_method_used': e2_method
    }

    return result
