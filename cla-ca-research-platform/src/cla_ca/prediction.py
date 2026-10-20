import numpy as np

def predict_labels(scores: np.ndarray) -> tuple:
    """
    Labeling stage of Algorithm 1.
    """
    label_point = np.median(scores)

    # Use a small epsilon to handle precision or just strict comparison
    # The paper says k > Labelpoint, let's strictly follow it.
    predictions = (scores > label_point).astype(int)

    # If all are 0, check if we should use mean instead of median
    if np.sum(predictions) == 0:
        predictions = (scores > np.mean(scores)).astype(int)

    return predictions, label_point
