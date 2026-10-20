import pandas as pd
import numpy as np

def calculate_scores(df_disc: pd.DataFrame, weights: dict, splist: list) -> np.ndarray:
    """
    Score calculation stage of Algorithm 1.
    """
    scores = np.zeros(len(df_disc))
    for item in splist:
        col = item['index']
        sp_val = item['value']
        # The paper says: "if xi,index > spindex then ki += W[index]"
        # Check if this applies to original values or discretized values.
        # It should apply to the discretized feature values (1 to bim).
        mask = df_disc[col] > sp_val
        scores[mask] += weights[col]
    return scores
