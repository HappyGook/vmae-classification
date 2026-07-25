import numpy as np


def compute_binary_metrics(y_true, y_pred):
    """Compute confusion matrix, accuracy, precision, recall, and F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {}

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / len(y_true)

    return {
        "n": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_window_metrics(windows):
    """Compute metrics from windows containing `true_label` and `pred_label`."""
    return compute_binary_metrics(
        [window["true_label"] for window in windows],
        [window["pred_label"] for window in windows],
    )