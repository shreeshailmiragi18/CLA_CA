import pandas as pd
import numpy as np

def discretize_dataset(df: pd.DataFrame, feature_cols: list, bim: int) -> pd.DataFrame:
    """
    Apply equal-width discretization.
    """
    df_disc = df.copy()
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        # Even if min==max, use bin 1
        if max_val == min_val:
            df_disc[col] = 1
            continue

        # Create bins
        bins = np.linspace(min_val, max_val, bim + 1)
        # Use pd.cut with include_lowest=True
        # For equal width, bins is best.
        df_disc[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True) + 1
        # If NaN, means outside bin (shouldn't happen with min/max), fill with 1
        df_disc[col] = df_disc[col].fillna(1).astype(int)

    return df_disc
