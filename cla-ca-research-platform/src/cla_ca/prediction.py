import numpy as np

def predict_labels(scores: np.ndarray) -> tuple:
    """
    Labeling stage of Algorithm 1.
    Paper: Ks = unique(Ks); Labelpoint = median(Ks)
    """
    # Use median of UNIQUE scores as per paper
    unique_scores = np.unique(scores)
    label_point = np.median(unique_scores)

    # Paper: if k > Labelpoint then defective else non-defective
    predictions = (scores > label_point).astype(int)

    return predictions, label_point
