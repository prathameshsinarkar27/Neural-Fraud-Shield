"""
models/explainability.py
==============================================================================
Shared explainability & evaluation utilities, used at both training time
(train.py) and serving time (backend/predictor.py).
==============================================================================
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, confusion_matrix, roc_curve,
    precision_recall_curve
)



def compute_metrics(y_true, y_pred, y_prob, name):
    """ROC-AUC, PR-AUC, recall/precision/F1, accuracy, FPR, and confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics = {
        'name': name,
        'roc_auc': round(float(roc_auc_score(y_true, y_prob)), 4),
        'pr_auc': round(float(average_precision_score(y_true, y_prob)), 4),
        'recall': round(float(recall_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, y_pred)), 4),
        'accuracy': round(float((tp + tn) / (tp + tn + fp + fn)), 4),
        'fpr': round(float(fpr_val), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    print(f"\n{name}:")
    for k, v in metrics.items():
        if k not in ('name', 'confusion_matrix'):
            print(f"  {k}: {v}")
    return metrics


def compute_curve(y_true, y_prob):
    """Downsampled ROC curve (fpr/tpr) for charting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    step = max(1, len(fpr) // 200)
    return {
        'fpr': [round(float(x), 4) for x in fpr[::step]],
        'tpr': [round(float(x), 4) for x in tpr[::step]]
    }


def compute_pr_curve(y_true, y_prob):
    """Downsampled Precision-Recall curve for charting."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    step = max(1, len(precision) // 200)
    return {
        'precision': [round(float(x), 4) for x in precision[::step]],
        'recall': [round(float(x), 4) for x in recall[::step]]
    }


def compute_threshold_data(y_true, y_prob):
    """Precision/recall/FPR/F1 swept across thresholds, for the interactive slider."""
    thresholds = np.arange(0.0, 1.01, 0.02)
    data = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        data.append({
            'threshold': round(float(t), 2),
            'precision': round(float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0, 4),
            'recall': round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
            'fpr': round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
            'f1': round(float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0, 4)
        })
    return data


