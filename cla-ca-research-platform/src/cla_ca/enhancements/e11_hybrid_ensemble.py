"""
E11: Hybrid Ensemble - Best of CLA-CA + ML + Advanced Thresholding

Combines:
1. CLA-CA scores (unsupervised, no label leakage)
2. Multiple ML classifiers trained with CV (supervised)
3. Advanced thresholding (Pareto, Quantile, Youden)
4. Stacking ensemble for final prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef


def run_e11_hybrid_ensemble(
    dataset_name: str,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Run E11 Hybrid Ensemble with 5-fold CV.
    Uses CLA-CA score as meta-feature + multiple classifiers + stacking.
    """
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
    y_mat = np.array(y_true)

    # 2. Stratified K-Fold CV with Stacking
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_preds_all = np.zeros(len(y_mat))
    y_probs_all = np.zeros(len(y_mat))

    for train_idx, test_idx in skf.split(X_aug, y_mat):
        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        y_train, y_test = y_mat[train_idx], y_mat[test_idx]

        # Base classifiers (diverse). Pipelines keep scaling inside each fold.
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=random_state, max_depth=10)
        et = ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=random_state, max_depth=10)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=3)
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000, C=0.1)
        )
        svm = make_pipeline(
            StandardScaler(),
            SVC(class_weight='balanced', probability=True, random_state=random_state, C=1.0, gamma='scale')
        )
        knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance'))

        # StackingClassifier creates out-of-fold meta-features, avoiding the
        # in-sample meta-training leakage in the previous implementation.
        base_clfs = [
            ('rf', rf), ('et', et), ('gb', gb),
            ('lr', lr), ('svm', svm), ('knn', knn)
        ]
        meta_clf = StackingClassifier(
            estimators=base_clfs,
            final_estimator=LogisticRegression(
                class_weight='balanced', random_state=random_state, max_iter=1000
            ),
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1,
        )
        meta_clf.fit(X_train, y_train)

        # Final ensemble predictions
        meta_probs = meta_clf.predict_proba(X_test)[:, 1]
        meta_preds = (meta_probs > 0.5).astype(int)

        y_preds_all[test_idx] = meta_preds
        y_probs_all[test_idx] = meta_probs

    # Also compute CLA-CA baseline for comparison
    from cla_ca.prediction import predict_labels
    y_pred_claca, _ = predict_labels(claca_scores)

    # Metrics
    precision = precision_score(y_mat, y_preds_all, zero_division=0)
    recall = recall_score(y_mat, y_preds_all, zero_division=0)
    f1 = f1_score(y_mat, y_preds_all, zero_division=0)
    mcc = matthews_corrcoef(y_mat, y_preds_all)
    g_mean = float(np.sqrt(recall * precision))

    # CLA-CA baseline metrics
    base_precision = precision_score(y_mat, y_pred_claca, zero_division=0)
    base_recall = recall_score(y_mat, y_pred_claca, zero_division=0)
    base_f1 = f1_score(y_mat, y_pred_claca, zero_division=0)
    base_mcc = matthews_corrcoef(y_mat, y_pred_claca)
    base_g_mean = float(np.sqrt(base_recall * base_precision))

    return {
        'dataset': dataset_name,
        'model': 'E11_Hybrid_Ensemble',
        'total_modules': len(y_mat),
        'actual_defects': int(sum(y_mat)),
        'predicted_defects_ensemble': int(sum(y_preds_all)),
        'predicted_defects_claca': int(sum(y_pred_claca)),
        'metrics_ensemble': {
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'mcc': float(mcc),
            'g_mean': float(g_mean)
        },
        'metrics_claca': {
            'recall': float(base_recall),
            'precision': float(base_precision),
            'f1': float(base_f1),
            'mcc': float(base_mcc),
            'g_mean': float(base_g_mean)
        }
    }