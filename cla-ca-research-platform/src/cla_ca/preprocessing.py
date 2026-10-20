import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_dataset(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Apply Z-score normalization to feature columns.
    """
    scaler = StandardScaler()
    df_norm = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_norm
