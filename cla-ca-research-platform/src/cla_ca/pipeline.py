import pandas as pd
import numpy as np
import os
import json

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
    Execute full CLA-CA pipeline on a given dataset.
    """
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    if bim is None:
        bim = 150 if is_aeeem else 35

    feature_cols = list(X_df.columns)
    full_df = X_df.copy()
    full_df['target'] = y_true

    # 1. Normalization & SetWeight
    df_norm = normalize_dataset(full_df, feature_cols)
    weights = set_weight(df_norm, feature_cols, alpha=alpha)

    # 2. Discretization
    df_disc = discretize_dataset(full_df, feature_cols, bim=bim)

    # 3. Feature Split (Createsplit)
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

    # 4. Module Scoring
    scores = calculate_scores(df_disc, weights, splist)

    # 5. Prediction
    y_pred, label_point = predict_labels(scores)

    # 6. Evaluation Metrics
    metrics = calculate_metrics(y_true, y_pred)

    result = {
        'dataset': dataset_name,
        'is_aeeem': is_aeeem,
        'bim': bim,
        'alpha': alpha,
        'total_features': len(feature_cols),
        'retained_features_count': len(retained_features),
        'excluded_features_count': len(excluded_features),
        'retained_features': retained_features,
        'excluded_features': excluded_features,
        'label_point': float(label_point),
        'metrics': metrics
    }
    return result

if __name__ == '__main__':
    os.makedirs('results/baseline', exist_ok=True)
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
                'Precision': res['metrics']['precision'],
                'F1': res['metrics']['f1'],
                'MCC': res['metrics']['mcc'],
                'G-mean': res['metrics']['g_mean'],
                'Retained Features': f"{res['retained_features_count']}/{res['total_features']}"
            })
            print(f"Finished {ds}: F1={res['metrics']['f1']:.4f}, MCC={res['metrics']['mcc']:.4f}")
        except Exception as e:
            print(f"Error on {ds}: {e}")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv('results/baseline/summary.csv', index=False)
    print("\n=== BASELINE SUMMARY ===")
    print(sum_df.to_string(index=False))
