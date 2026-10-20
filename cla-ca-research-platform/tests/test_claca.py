import pytest
import numpy as np
import pandas as pd
from cla_ca.preprocessing import normalize_dataset
from cla_ca.feature_weighting import set_weight
from cla_ca.discretization import discretize_dataset
from cla_ca.valley_detection import create_split
from cla_ca.scoring import calculate_scores
from cla_ca.prediction import predict_labels
from cla_ca.metrics import calculate_metrics

def test_preprocessing():
    df = pd.DataFrame({'f1': [1.0, 2.0, 3.0], 'f2': [10.0, 20.0, 30.0]})
    df_norm = normalize_dataset(df, ['f1', 'f2'])
    assert np.isclose(df_norm['f1'].mean(), 0.0)

def test_feature_weighting():
    df = pd.DataFrame({'f1': [1.0, 2.0, 10.0], 'f2': [5.0, 5.1, 5.2]})
    weights = set_weight(df, ['f1', 'f2'], alpha=0.8)
    assert 'f1' in weights
    assert 'f2' in weights

def test_discretization():
    df = pd.DataFrame({'f1': [0.0, 50.0, 100.0]})
    df_disc = discretize_dataset(df, ['f1'], bim=5)
    assert df_disc['f1'].min() >= 1
    assert df_disc['f1'].max() <= 5

def test_valley_detection():
    # Synthetic counts creating a valley at index 2 (1-based: 3)
    vals = np.array([1, 1, 2, 3, 3, 3, 2, 1, 1])
    sp = create_split(vals, bim=5)
    # Just check it returns an integer
    assert isinstance(sp, int)

def test_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = calculate_metrics(y_true, y_pred)
    assert 'f1' in m
    assert 'mcc' in m
