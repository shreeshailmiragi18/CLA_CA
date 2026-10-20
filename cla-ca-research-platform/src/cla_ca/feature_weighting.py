import numpy as np
import pandas as pd

def set_weight(df_norm: pd.DataFrame, feature_cols: list, alpha: float = 0.8) -> dict:
    """
    Calculate feature weights based on Algorithm 2 in the paper.
    """
    vars_arr = df_norm[feature_cols].var().values
    max_var = np.max(vars_arr)
    weights = {}
    e = np.e

    for i, col in enumerate(feature_cols):
        v = vars_arr[i]
        if v == 0:
            weights[col] = e
            continue

        ratio = max_var / v
        if ratio > 1.5:
            weights[col] = e
        else:
            weights[col] = e + alpha * (v / max_var)

    return weights
