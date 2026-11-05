"""E13: leakage-safe calibrated ensemble for defect prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _build_ensemble(random_state: int) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "logistic",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight="balanced",
                        C=0.25,
                        max_iter=1500,
                        random_state=random_state,
                    ),
                ),
            ),
            (
                "svm",
                make_pipeline(
                    StandardScaler(),
                    SVC(
                        class_weight="balanced",
                        C=0.75,
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ),
        ],
        voting="soft",
        weights=[2, 2, 1, 1],
        n_jobs=-1,
    )


def _select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(
        np.quantile(probabilities, np.linspace(0.05, 0.95, 91))
    )
    best_threshold = 0.5
    best_key = (-1.0, -1.0)

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        key = (
            f1_score(labels, predictions, zero_division=0),
            matthews_corrcoef(labels, predictions),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)

    return best_threshold


def _claca_feature(dataset_name: str, alpha: float, bim: int | None) -> tuple:
    from cla_ca.data_loader import load_dataset

    features, labels, is_aeeem = load_dataset(dataset_name)
    return _claca_feature_from_data(features, labels, is_aeeem, alpha, bim)


def _claca_feature_from_data(
    features: pd.DataFrame,
    labels: np.ndarray,
    is_aeeem: bool,
    alpha: float,
    bim: int | None,
) -> tuple:
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.discretization import discretize_dataset
    from cla_ca.valley_detection import create_split
    from cla_ca.scoring import calculate_scores

    if bim is None:
        bim = 155 if is_aeeem else 40

    feature_cols = list(features.columns)
    full_df = features.copy()
    full_df["target"] = labels
    normalized = normalize_dataset(full_df, feature_cols)
    weights = set_weight(normalized, feature_cols, alpha=alpha)
    discretized = discretize_dataset(normalized, feature_cols, bim=bim)
    split_points = []
    for column in feature_cols:
        split = create_split(discretized[column].values, bim=bim)
        if split > 0:
            split_points.append({"index": column, "value": split})

    scores = calculate_scores(discretized, weights, split_points)
    ranks = pd.Series(scores).rank(method="average", pct=True).to_numpy()
    matrix = np.column_stack([normalized[feature_cols].to_numpy(), ranks])
    return matrix, np.asarray(labels, dtype=int), is_aeeem


def _run_e13_on_features(
    X: np.ndarray,
    y: np.ndarray,
    is_aeeem: bool,
    dataset_name: str,
    n_splits: int,
    random_state: int,
) -> dict:
    """Run the shared E13 cross-validation procedure on prepared features."""
    minority_count = int(np.bincount(y).min())
    outer_splits = min(n_splits, minority_count)
    if outer_splits < 2:
        raise ValueError(f"{dataset_name} needs at least two samples in each class")

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    predictions = np.zeros(len(y), dtype=int)
    probabilities = np.zeros(len(y), dtype=float)
    thresholds = []

    for train_idx, test_idx in outer_cv.split(X, y):
        y_train = y[train_idx]
        inner_splits = min(3, int(np.bincount(y_train).min()))
        inner_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )
        calibration_model = _build_ensemble(random_state)
        calibration_probabilities = cross_val_predict(
            calibration_model,
            X[train_idx],
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = _select_threshold(calibration_probabilities, y_train)

        model = _build_ensemble(random_state)
        model.fit(X[train_idx], y_train)
        test_probabilities = model.predict_proba(X[test_idx])[:, 1]
        probabilities[test_idx] = test_probabilities
        predictions[test_idx] = (test_probabilities >= threshold).astype(int)
        thresholds.append(threshold)

    from cla_ca.metrics import calculate_metrics

    metrics = calculate_metrics(y, predictions)
    return {
        "dataset": dataset_name,
        "model": "E13_Calibrated_Ensemble",
        "is_aeeem": is_aeeem,
        "total_modules": int(len(y)),
        "actual_defects": int(y.sum()),
        "predicted_defects": int(predictions.sum()),
        "threshold_mean": float(np.mean(thresholds)),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "predictions": predictions,
        "probabilities": probabilities,
    }


def run_e13_calibrated_ensemble(
    dataset_name: str,
    n_splits: int = 5,
    alpha: float = 0.8,
    bim: int | None = None,
    random_state: int = 42,
) -> dict:
    """Evaluate E13 with nested threshold calibration and outer CV predictions."""
    X, y, is_aeeem = _claca_feature(dataset_name, alpha, bim)
    result = _run_e13_on_features(X, y, is_aeeem, dataset_name, n_splits, random_state)
    result.pop("predictions")
    result.pop("probabilities")
    return result


def run_e13_calibrated_ensemble_from_data(
    features: pd.DataFrame,
    labels: np.ndarray,
    is_aeeem: bool = False,
    dataset_name: str = "uploaded",
    n_splits: int = 5,
    alpha: float = 0.8,
    bim: int | None = None,
    random_state: int = 42,
) -> dict:
    """Run E13 on an uploaded dataframe and return predictions for the API."""
    X, y, is_aeeem = _claca_feature_from_data(features, labels, is_aeeem, alpha, bim)
    return _run_e13_on_features(X, y, is_aeeem, dataset_name, n_splits, random_state)
