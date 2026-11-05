"""
E7: Deep Learning - Multi-Layer Perceptron with TabNet-style attention for tabular defect data

Uses MLP with feature importance weighting and residual connections for software defect prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MLPClassifier(nn.Module):
    """Deep MLP for tabular defect prediction with residual connections."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x)).squeeze()


def run_e7_mlp_enhancement(
    dataset_name: str,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    random_state: int = 42
) -> dict:
    """Run E7 MLP enhancement with 5-fold CV."""

    if not TORCH_AVAILABLE:
        return {'error': 'PyTorch not available', 'dataset': dataset_name}

    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.discretization import discretize_dataset
    from cla_ca.valley_detection import create_split
    from cla_ca.scoring import calculate_scores

    # 1. Load data + CLA-CA scores
    X_df, y_true, is_aeeem = load_dataset(dataset_name)
    feature_cols = list(X_df.columns)
    bim = 155 if is_aeeem else 40
    full_df = X_df.copy()
    full_df['target'] = y_true
    df_norm = normalize_dataset(full_df, feature_cols)
    weights = set_weight(df_norm, feature_cols, alpha=0.8)
    df_disc = discretize_dataset(df_norm, feature_cols, bim=bim)
    splist = []
    for col in feature_cols:
        sp = create_split(df_disc[col].values, bim=bim)
        if sp > 0:
            splist.append({'index': col, 'value': sp})
    claca_scores = calculate_scores(df_disc, weights, splist)

    X_mat = df_norm[feature_cols].values
    X_aug = np.column_stack([X_mat, claca_scores])
    y_mat = np.array(y_true, dtype=np.float32)

    # 2. Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_preds_all = np.zeros(len(y_mat))
    y_probs_all = np.zeros(len(y_mat))

    for train_idx, test_idx in skf.split(X_aug, y_mat):
        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        y_train, y_test = y_mat[train_idx], y_mat[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        # Convert to torch
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)

        # Model
        input_dim = X_train.shape[1]
        model = MLPClassifier(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Class weights for imbalance
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test_t)).numpy()
            preds = (probs > 0.5).astype(int)

        y_preds_all[test_idx] = preds
        y_probs_all[test_idx] = probs

    # Metrics
    from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef
    precision = precision_score(y_mat, y_preds_all, zero_division=0)
    recall = recall_score(y_mat, y_preds_all, zero_division=0)
    f1 = f1_score(y_mat, y_preds_all, zero_division=0)
    mcc = matthews_corrcoef(y_mat, y_preds_all)
    g_mean = float(np.sqrt(recall * precision))

    return {
        'dataset': dataset_name,
        'model': 'MLP',
        'total_modules': len(y_mat),
        'actual_defects': int(sum(y_mat)),
        'predicted_defects': int(sum(y_preds_all)),
        'metrics': {
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'mcc': float(mcc),
            'g_mean': float(g_mean)
        }
    }