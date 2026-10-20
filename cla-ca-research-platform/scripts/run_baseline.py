import pandas as pd
import numpy as np
import os

from cla_ca.data_loader import load_dataset
from cla_ca.preprocessing import normalize_dataset
from cla_ca.feature_weighting import set_weight
from cla_ca.discretization import discretize_dataset
from cla_ca.valley_detection import create_split
from cla_ca.scoring import calculate_scores
from cla_ca.prediction import predict_labels
from cla_ca.metrics import calculate_metrics

def run_claca_pipeline(dataset_name: str, bim: int = None, alpha: float = 0.8) -> dict:
    """
    Execute full CLA-CA pipeline.
    """
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    if bim is None:
        bim = 150 if is_aeeem else 35

    feature_cols = list(X_df.columns)

    # 1. Normalization & SetWeight (on normalized data)
    df_norm = normalize_dataset(X_df, feature_cols)
    weights = set_weight(df_norm, feature_cols, alpha=alpha)

    # 2. Discretization (on original data)
    df_disc = discretize_dataset(X_df, feature_cols, bim=bim)

    # 3. Feature Split (Createsplit on discretized data)
    splist = []
    retained_features = []

    for col in feature_cols:
        attr_vals = df_disc[col].values
        sp = create_split(attr_vals, bim=bim)
        if sp > 0:
            splist.append({'index': col, 'value': sp})
            retained_features.append(col)

    # 4. Module Scoring (on discretized data)
    scores = calculate_scores(df_disc, weights, splist)

    # 5. Prediction
    y_pred, label_point = predict_labels(scores)

    # 6. Evaluation Metrics
    metrics = calculate_metrics(y_true, y_pred)

    result = {
        'dataset': dataset_name,
        'metrics': metrics
    }
    return result

if __name__ == '__main__':
    datasets = [
        'ant-1.5', 'arc', 'berek', 'camel-1.0', 'ivy-2.0',
        'jedit-3.2', 'synapse-1.1', 'tomcat', 'xalan-2.4', 'xerces-1.2',
        'EQ', 'JDT', 'LC', 'ML', 'PDE'
    ]

    summary = []
    for ds in datasets:
        try:
            res = run_claca_pipeline(ds)
            summary.append({
                'Dataset': ds,
                'Recall': res['metrics']['recall'],
                'F1': res['metrics']['f1'],
                'MCC': res['metrics']['mcc'],
                'G-mean': res['metrics']['g_mean']
            })
        except Exception as e:
            print(f"Error on {ds}: {e}")

    sum_df = pd.DataFrame(summary)
    print("\n=== FINAL BASELINE RESULTS ===")
    print(sum_df.to_string(index=False))
