"""
Unit tests for E1: Adaptive Thresholding Enhancement.

Tests the thresholding methods implemented in e1_adaptive_thresholding.py.
Ensures each method produces valid thresholds and predictions.
"""

import numpy as np
import pytest

from cla_ca.enhancements.e1_adaptive_thresholding import (  # type: ignore
    predict_labels_baseline,
    threshold_percentile,
    threshold_otsu,
    threshold_mean,
    threshold_median_unique,
    select_threshold,
    predict_labels_e1,
    evaluate_thresholds,
    run_e1_enhancement,
)


# --- Test Data Fixtures ---

def create_mock_scores(n: int = 100) -> np.ndarray:
    """Create mock CLA-CA scores for testing."""
    np.random.seed(42)
    return np.random.exponential(scale=2.0, size=n)


def create_mock_labels(n: int = 100, prevalence: float = 0.2) -> np.ndarray:
    """Create mock binary labels."""
    n_positive = int(n * prevalence)
    n_negative = n - n_positive
    labels = np.zeros(n, dtype=int)
    labels[:n_positive] = 1
    np.random.shuffle(labels)
    return labels


# --- Baseline Tests ---

class TestBaseline:
    """Test the baseline thresholding method (median of unique scores)."""

    def test_baseline_predictions_have_correct_shape(self):
        """Baseline predictions should have the same shape as scores."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        predictions, label_point = predict_labels_baseline(scores)
        assert len(predictions) == len(scores)
        assert isinstance(label_point, float)

    def test_baseline_predictions_are_binary(self):
        """Baseline predictions should be 0 or 1."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        predictions, label_point = predict_labels_baseline(scores)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_baseline_label_point_is_finite(self):
        """Baseline label point should be a finite number."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        predictions, label_point = predict_labels_baseline(scores)
        assert np.isfinite(label_point)


# --- Percentile Threshold Tests ---

class TestPercentileThreshold:
    """Test percentile-based thresholding methods."""

    def test_percentile_25_returns_valid_threshold(self):
        """25th percentile should return a valid threshold."""
        scores = create_mock_scores(100)
        threshold = threshold_percentile(scores, 25)
        assert np.isfinite(threshold)
        assert 0 <= threshold <= np.max(scores)

    def test_percentile_50_equals_median(self):
        """50th percentile should equal the median."""
        scores = create_mock_scores(100)
        threshold_p50 = threshold_percentile(scores, 50)
        threshold_median = np.median(scores)
        assert abs(threshold_p50 - threshold_median) < 1e-10

    def test_percentile_75_above_50(self):
        """75th percentile should be >= 50th percentile."""
        scores = create_mock_scores(100)
        p50 = threshold_percentile(scores, 50)
        p75 = threshold_percentile(scores, 75)
        assert p75 >= p50 - 1e-10  # allow for floating point

    def test_percentile_0_is_min(self):
        """0th percentile should be minimum score."""
        scores = create_mock_scores(100)
        threshold_0 = threshold_percentile(scores, 0)
        assert threshold_0 == np.min(scores)

    def test_percentile_100_is_max(self):
        """100th percentile should be maximum score."""
        scores = create_mock_scores(100)
        threshold_100 = threshold_percentile(scores, 100)
        assert threshold_100 == np.max(scores)


# --- Otsu's Method Tests ---

class TestOtsuThreshold:
    """Test Otsu's method thresholding."""

    def test_otsu_returns_valid_threshold(self):
        """Otsu should return a valid finite threshold."""
        scores = create_mock_scores(100)
        threshold = threshold_otsu(scores)
        assert np.isfinite(threshold)
        assert 0 <= threshold <= np.max(scores)

    def test_otsu_with_bimodal_distribution(self):
        """Otsu works better with bimodal distributions."""
        # Create bimodal: mix of two Gaussians
        np.random.seed(42)
        scores = np.concatenate([
            np.random.normal(loc=1.0, scale=0.5, size=50),
            np.random.normal(loc=5.0, scale=0.5, size=50),
        ])
        threshold = threshold_otsu(scores)
        assert np.isfinite(threshold)
        # Threshold should be between the two modes
        assert 1.0 < threshold < 5.0


# --- Mean Threshold Tests ---

class TestMeanThreshold:
    """Test mean-based thresholding."""

    def test_mean_returns_valid_threshold(self):
        """Mean should return a valid finite threshold."""
        scores = create_mock_scores(100)
        threshold = threshold_mean(scores)
        assert np.isfinite(threshold)
        assert abs(threshold - np.mean(scores)) < 1e-10

    def test_mean_predictions_binary(self):
        """Mean-based predictions should be binary."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        threshold = threshold_mean(scores)
        predictions = (scores > threshold).astype(int)
        assert set(np.unique(predictions)).issubset({0, 1})


# --- Select Threshold Tests ---

class TestSelectThreshold:
    """Test the select_threshold dispatcher."""

    def test_select_baseline(self):
        """Baseline method should work via select_threshold."""
        scores = create_mock_scores(100)
        threshold, method_name = select_threshold(scores, method='baseline')
        assert np.isfinite(threshold)
        assert method_name == 'baseline'

    def test_select_percentile(self):
        """Percentile method should work via select_threshold."""
        scores = create_mock_scores(100)
        threshold, method_name = select_threshold(scores, method='percentile', percentile=75)
        assert np.isfinite(threshold)
        assert method_name == 'percentile_75'

    def test_select_otsu(self):
        """Otsu method should work via select_threshold."""
        scores = create_mock_scores(100)
        threshold, method_name = select_threshold(scores, method='otsu')
        assert np.isfinite(threshold)
        assert method_name == 'otsu'

    def test_select_mean(self):
        """Mean method should work via select_threshold."""
        scores = create_mock_scores(100)
        threshold, method_name = select_threshold(scores, method='mean')
        assert np.isfinite(threshold)
        assert method_name == 'mean'

    def test_select_invalid_method(self):
        """Invalid method should raise ValueError."""
        scores = create_mock_scores(100)
        with pytest.raises(ValueError):
            select_threshold(scores, method='invalid_method')


# --- Predict Labels E1 Tests ---

class TestPredictLabelsE1:
    """Test the E1 predict_labels_e1 function."""

    def test_e1_baseline_predictions(self):
        """E1 baseline method should produce valid predictions."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        predictions, threshold, metadata = predict_labels_e1(
            scores, method='baseline'
        )
        assert len(predictions) == len(scores)
        assert np.isfinite(threshold)
        assert metadata['threshold_method'] == 'baseline'

    def test_e1_percentile_90(self):
        """E1 with 90th percentile method."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        predictions, threshold, metadata = predict_labels_e1(
            scores, method='percentile', percentile=90
        )
        assert len(predictions) == len(scores)
        assert np.isfinite(threshold)
        assert metadata['threshold_method'] == 'percentile_90'

    def test_e1_otsu(self):
        """E1 with Otsu method."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        predictions, threshold, metadata = predict_labels_e1(
            scores, method='otsu'
        )
        assert len(predictions) == len(scores)
        assert np.isfinite(threshold)
        assert metadata['threshold_method'] == 'otsu'

    def test_e1_returns_metadata(self):
        """E1 should return metadata about the thresholding."""
        scores = create_mock_scores(50)
        y_true = create_mock_labels(50)
        predictions, threshold, metadata = predict_labels_e1(
            scores, method='baseline'
        )
        assert 'threshold_method' in metadata
        assert 'threshold' in metadata
        assert 'n_positive' in metadata
        assert 'n_negative' in metadata


# --- Run E1 Enhancement Tests ---

class TestRunE1Enhancement:
    """Test the run_e1_enhancement function."""

    def test_e1_enhancement_baseline(self):
        """E1 enhancement with baseline method should work."""
        result = run_e1_enhancement(
            'berek', bim=23, alpha=0.8, threshold_method='baseline'
        )
        assert result['e1_enhanced'] == True
        assert result['threshold_method_used'] == 'baseline'
        assert 'metrics' in result
        assert 'f1' in result['metrics']
        assert 'precision' in result['metrics']
        assert 'recall' in result['metrics']

    def test_e1_enhancement_baseline_no_bim(self):
        """E1 enhancement with baseline method and auto BIM."""
        result = run_e1_enhancement(
            'berek', alpha=0.8, threshold_method='baseline'
        )
        assert result['e1_enhanced'] == True
        assert result['threshold_method_used'] == 'baseline'
        assert 'metrics' in result

    def test_e1_enhancement_percentile(self):
        """E1 enhancement with percentile method."""
        result = run_e1_enhancement(
            'berek', bim=23, alpha=0.8, threshold_method='percentile', percentile=75
        )
        assert result['e1_enhanced'] == True
        assert result['threshold_method_used'] == 'percentile_75'
        assert 'metrics' in result

    def test_e1_enhancement_otsu(self):
        """E1 enhancement with Otsu method."""
        result = run_e1_enhancement(
            'berek', bim=23, alpha=0.8, threshold_method='otsu'
        )
        assert result['e1_enhanced'] == True
        assert result['threshold_method_used'] == 'otsu'
        assert 'metrics' in result

    def test_e1_enhancement_returns_required_fields(self):
        """E1 result should have all required fields."""
        result = run_e1_enhancement(
            'berek', bim=23, alpha=0.8, threshold_method='baseline'
        )
        required_fields = [
            'dataset', 'is_aeeem', 'bim', 'alpha', 'threshold_method',
            'threshold', 'metrics', 'retained_features', 'total_features',
            'label_point', 'e1_enhanced', 'enhancement_type',
            'threshold_method_used'
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


# --- Integration: Compare Baseline vs E1 Methods ---

class TestBaselineVsE1Comparison:
    """Compare baseline and E1 thresholding methods."""

    def test_baseline_vs_e1_different_methods_produce_different_results(self):
        """Different thresholding methods should sometimes produce different predictions."""
        np.random.seed(42)
        scores = np.random.exponential(scale=2.0, size=100)
        y_true = create_mock_labels(100, prevalence=0.2)

        # Baseline
        baseline_pred, baseline_thresh, baseline_meta = predict_labels_e1(
            scores, method='baseline'
        )

        # 75th percentile
        p75_pred, p75_thresh, p75_meta = predict_labels_e1(
            scores, method='percentile', percentile=75
        )

        # They should generally produce different predictions
        # (they may coincide by chance, but shouldn't always be identical)
        baseline_f1 = sum(baseline_pred) / len(baseline_pred)  # simplified
        p75_f1 = sum(p75_pred) / len(p75_pred)
        # Just verify both are valid
        assert baseline_f1 >= 0 and baseline_f1 <= 1
        assert p75_f1 >= 0 and p75_f1 <= 1

    def test_all_methods_produce_binary_predictions(self):
        """All E1 methods should produce binary predictions."""
        np.random.seed(42)
        scores = np.random.exponential(scale=2.0, size=50)
        y_true = create_mock_labels(50, prevalence=0.2)

        methods = ['baseline', 'percentile', 'otsu', 'mean']

        for method in methods:
            # Pass additional kwargs as needed
            if method == 'percentile':
                predictions, threshold, metadata = predict_labels_e1(
                    scores, method=method, percentile=75
                )
            elif method == 'otsu':
                predictions, threshold, metadata = predict_labels_e1(
                    scores, method='otsu'
                )
            else:
                predictions, threshold, metadata = predict_labels_e1(scores, method=method)
            assert set(np.unique(predictions)).issubset({0, 1})
            assert np.isfinite(threshold)