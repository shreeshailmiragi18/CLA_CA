"""
E13: Extended Dataset Evaluation - Adding NASA MDP and additional datasets

NASA MDP datasets (classic defect prediction benchmarks):
- CM1, KC1, KC2, KC3, MC1, MC2, MW1, PC1, PC2, PC3, PC4, PC5
- These are widely used in defect prediction literature

We'll download and prepare these for comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import os


NASA_DATASETS = {
    'cm1': 'https://github.com/klainfo/NASAMetricsData/raw/master/CM1.arff',
    'kc1': 'https://github.com/klainfo/NASAMetricsData/raw/master/KC1.arff',
    'kc2': 'https://github.com/klainfo/NASAMetricsData/raw/master/KC2.arff',
    'kc3': 'https://github.com/klainfo/NASAMetricsData/raw/master/KC3.arff',
    'mc1': 'https://github.com/klainfo/NASAMetricsData/raw/master/MC1.arff',
    'mc2': 'https://github.com/klainfo/NASAMetricsData/raw/master/MC2.arff',
    'mw1': 'https://github.com/klainfo/NASAMetricsData/raw/master/MW1.arff',
    'pc1': 'https://github.com/klainfo/NASAMetricsData/raw/master/PC1.arff',
    'pc2': 'https://github.com/klainfo/NASAMetricsData/raw/master/PC2.arff',
    'pc3': 'https://github.com/klainfo/NASAMetricsData/raw/master/PC3.arff',
    'pc4': 'https://github.com/klainfo/NASAMetricsData/raw/master/PC4.arff',
    'pc5': 'https://github.com/klainfo/NASAMetricsData/raw/master/PC5.arff',
}


def download_nasa_datasets(data_dir: str = '/Users/shreeshailmiragi/Desktop/claude/dataset/NASA'):
    """Download NASA MDP datasets."""
    os.makedirs(data_dir, exist_ok=True)

    for name, url in NASA_DATASETS.items():
        filepath = os.path.join(data_dir, f'{name}.arff')
        if not os.path.exists(filepath):
            print(f'Downloading {name}...')
            try:
                urlretrieve(url, filepath)
                print(f'  Saved to {filepath}')
            except Exception as e:
                print(f'  Failed: {e}')
        else:
            print(f'{name} already exists')


def parse_arff(filepath: str) -> pd.DataFrame:
    """Parse ARFF file to DataFrame."""
    from scipy.io import arff
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    # Convert byte strings to regular strings
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.decode('utf-8')
    return df


def prepare_nasa_dataset(name: str, data_dir: str = '/Users/shreeshailmiragi/Desktop/claude/dataset/NASA') -> tuple:
    """Prepare NASA dataset for CLA-CA pipeline."""
    filepath = os.path.join(data_dir, f'{name}.arff')
    df = parse_arff(filepath)

    # NASA datasets typically have 'defects' as target (number of defects)
    # Convert to binary: 0 = clean, >0 = defective
    if 'defects' in df.columns:
        y = (df['defects'] > 0).astype(int).values
        X = df.drop('defects', axis=1)
    elif 'defect' in df.columns:
        y = (df['defect'] > 0).astype(int).values
        X = df.drop('defect', axis=1)
    else:
        # Try last column
        y = (df.iloc[:, -1] > 0).astype(int).values
        X = df.iloc[:, :-1]

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    # Remove constant columns
    X = X.loc[:, X.var() > 0]

    # Handle missing values
    X = X.fillna(X.median())

    return X, y


def get_nasa_dataset_info():
    """Get info about all NASA datasets."""
    info = {}
    for name in NASA_DATASETS.keys():
        filepath = f'/Users/shreeshailmiragi/Desktop/claude/dataset/NASA/{name}.arff'
        if os.path.exists(filepath):
            df = parse_arff(filepath)
            if 'defects' in df.columns:
                y = (df['defects'] > 0).astype(int)
            elif 'defect' in df.columns:
                y = (df['defect'] > 0).astype(int)
            else:
                y = (df.iloc[:, -1] > 0).astype(int)

            info[name.upper()] = {
                'modules': len(df),
                'features': len(df.columns) - 1,
                'defective': int(y.sum()),
                'defect_rate': float(y.mean())
            }
    return info


if __name__ == '__main__':
    download_nasa_datasets()
    print("\nNASA Dataset Info:")
    info = get_nasa_dataset_info()
    for name, stats in info.items():
        print(f"  {name}: {stats['modules']} modules, {stats['features']} features, {stats['defective']} defective ({stats['defect_rate']:.1%})")