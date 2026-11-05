import pandas as pd
import numpy as np

def calculate_scores(df_disc: pd.DataFrame, weights: dict, splist: list) -> np.ndarray:
    """
    Score calculation stage of Algorithm 1.
    Paper: If xi,index > spindex then ki += W[index]
    The discretized regions are 1 to bim. The split point is a region index.
    If the discretized value is > split point (strictly), it means it's after the split region.
    """
    scores = np.zeros(len(df_disc))
    for item in splist:
        col = item['index']
        sp_val = item['value']
        # Paper says: if xi,index > spindex (strictly greater)
        mask = df_disc[col] > sp_val
        scores[mask] += weights[col]
    return scores
