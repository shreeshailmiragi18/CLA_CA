import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_dataset(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Apply Min-Max normalization to feature columns (as used in CLA-CA paper).
    """
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_norm
