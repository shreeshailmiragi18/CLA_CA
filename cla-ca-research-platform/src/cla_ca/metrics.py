import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate performance measures defined in Section V.B.
    """
    # Confusion Matrix:
    # rows: true, cols: predicted
    # Labels 0, 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    # G-mean = sqrt(precision * recall)
    g_mean = np.sqrt(precision * recall)

    return {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'g_mean': g_mean
    }
