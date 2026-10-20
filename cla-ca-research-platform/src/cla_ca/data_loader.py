import pandas as pd
import numpy as np
import os

def load_dataset(dataset_name: str) -> tuple:
    """
    Load dataset from PROMISE or AEEEM directory.
    Identifies feature columns and target label column.
    """
    # Path logic: The datasets are in the top-level 'dataset' folder
    # Assuming this script is run from project root, we need to go up one level.
    # Actually, let's just look in the workspace root.

    workspace_root = '/Users/shreeshailmiragi/Desktop/claude'
    dataset_base = os.path.join(workspace_root, 'dataset')

    # Check PROMISE
    promise_path = os.path.join(dataset_base, 'PROMISE', f'{dataset_name}.csv')
    if os.path.exists(promise_path):
        df = pd.read_csv(promise_path)
        is_aeeem = False
    else:
        alt_path = os.path.join(dataset_base, 'PROMISE', f'data_{dataset_name}.csv')
        if os.path.exists(alt_path):
            df = pd.read_csv(alt_path)
            is_aeeem = False
        else:
            aeeem_path = os.path.join(dataset_base, 'AEEEM', f'{dataset_name}.csv')
            if os.path.exists(aeeem_path):
                df = pd.read_csv(aeeem_path)
                is_aeeem = True
            else:
                raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_base}")

    # Identify target column
    target_col = None
    for col in ['bug', 'defects', 'isDefective', 'class', 'Defective', 'buggy']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    # Convert target to 0 (clean) and 1 (buggy/defective)
    y = df[target_col]
    if y.dtype == object:
        y = y.apply(lambda val: 1 if str(val).strip().lower() in ['buggy', 'y', 'true', '1', 'yes', 'defective'] else 0)
    else:
        y = y.apply(lambda val: 1 if val > 0 else 0)

    # Filter features: drop target, drop string identifiers
    feature_cols = []
    # Identify non-numeric columns
    for c in df.columns:
        if c == target_col: continue
        # Force numeric conversion
        df[c] = pd.to_numeric(df[c], errors='coerce')
        # Only keep if column has data
        if df[c].notna().sum() > 0:
            feature_cols.append(c)

    df = df[feature_cols].fillna(df[feature_cols].median())

    return df, y.values, is_aeeem
