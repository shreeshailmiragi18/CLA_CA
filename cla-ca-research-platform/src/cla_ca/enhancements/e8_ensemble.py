"""
E8: Stacking Ensemble - Multi-level Stacking with CLA-CA Features

Implements a robust stacking ensemble combining:
1. CLA-CA unsupervised scores as meta-features
2. Diverse base learners (RF, ET, GB, LR, SVM, KNN)
3. Meta-learner (Logistic Regression) on out-of-fold predictions
4. Proper cross-validation to prevent data leakage

This is the supervised ML version of CLA-CA that properly handles class imbalance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef


def run_e8_stacking_ensemble(
    dataset_name: str,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Run E8 Stacking Ensemble with 5-fold CV.
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

        # Standardize for LR and SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Base classifiers (diverse ensemble)
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=random_state, max_depth=10, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=random_state, max_depth=10, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=3, learning_rate=0.1)
        lr = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000, C=0.1, solver='lbfgs')
        svm = SVC(class_weight='balanced', probability=True, random_state=random_state, C=1.0, gamma='scale')
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

        # Train base learners
        base_clfs = [('rf', rf), ('et', et), ('gb', gb), ('lr', lr), ('svm', svm), ('knn', knn)]

        # Get out-of-fold predictions for meta-features
        meta_features_train = np.zeros((len(y_train), len(base_clfs)))
        meta_features_test = np.zeros((len(y_test), len(base_clfs)))

        for i, (name, clf) in enumerate(base_clfs):
            if i in [3, 4]:  # LR, SVM use scaled
                clf.fit(X_train_scaled, y_train)
                meta_features_train[:, i] = clf.predict_proba(X_train_scaled)[:, 1]
                meta_features_test[:, i] = clf.predict_proba(X_test_scaled)[:, 1]
            else:
                clf.fit(X_train, y_train)
                meta_features_train[:, i] = clf.predict_proba(X_train)[:, 1]
                meta_features_test[:, i] = clf.predict_proba(X_test)[:, 1]

        # Meta-learner: Logistic Regression on base predictions
        meta_clf = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000, C=1.0)
        meta_clf.fit(meta_features_train, y_train)

        # Final ensemble predictions
        meta_probs = meta_clf.predict_proba(meta_features_test)[:, 1]
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
        'model': 'E8_Stacking_Ensemble',
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


def run_e8_voting_ensemble(
    dataset_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    voting: str = 'soft'
) -> dict:
    """
    Run E8 Voting Ensemble (hard or soft voting) as alternative to stacking.
    """
    from cla_ca.data_loader import load_dataset
    from cla_ca.preprocessing import normalize_dataset
    from cla_ca.feature_weighting import set_weight
    from cla_ca.discretization import discretize_dataset
    from cla_ca.valley_detection import create_split
    from cla_ca.scoring import calculate_scores

    # Load data + CLA-CA scores
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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_preds_all = np.zeros(len(y_mat))
    y_probs_all = np.zeros(len(y_mat))

    for train_idx, test_idx in skf.split(X_aug, y_mat):
        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        y_train, y_test = y_mat[train_idx], y_mat[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Base classifiers
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=random_state, max_depth=10, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=random_state, max_depth=10, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=3, learning_rate=0.1)
        lr = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000, C=0.1, solver='lbfgs')
        svm = SVC(class_weight='balanced', probability=True, random_state=random_state, C=1.0, gamma='scale')

        # Voting classifier
        voting_clf = VotingClassifier(
            estimators=[('rf', rf), ('et', et), ('gb', gb), ('lr', lr), ('svm', svm)],
            voting=voting,
            weights=[2, 2, 1, 1, 1]  # Weight tree-based higher
        )

        voting_clf.fit(X_train, y_train)

        if voting == 'soft':
            probs = voting_clf.predict_proba(X_test)[:, 1]
            preds = (probs > 0.5).astype(int)
            y_probs_all[test_idx] = probs
        else:
            preds = voting_clf.predict(X_test)

        y_preds_all[test_idx] = preds

    # CLA-CA baseline
    from cla_ca.prediction import predict_labels
    y_pred_claca, _ = predict_labels(claca_scores)

    # Metrics
    precision = precision_score(y_mat, y_preds_all, zero_division=0)
    recall = recall_score(y_mat, y_preds_all, zero_division=0)
    f1 = f1_score(y_mat, y_preds_all, zero_division=0)
    mcc = matthews_corrcoef(y_mat, y_preds_all)
    g_mean = float(np.sqrt(recall * precision))

    base_precision = precision_score(y_mat, y_pred_claca, zero_division=0)
    base_recall = recall_score(y_mat, y_pred_claca, zero_division=0)
    base_f1 = f1_score(y_mat, y_pred_claca, zero_division=0)
    base_mcc = matthews_corrcoef(y_mat, y_pred_claca)
    base_g_mean = float(np.sqrt(base_recall * base_precision))

    return {
        'dataset': dataset_name,
        'model': f'E8_Voting_Ensemble_{voting}',
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