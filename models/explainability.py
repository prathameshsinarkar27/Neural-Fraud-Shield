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

# ==================================== METRICS ====================================


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


# ================================= SHAP FITTING ===========================


def fit_shap_kernel_explainer(dnn_model, X_train_smote, background_size=100):
    """Fit a SHAP KernelExplainer against the DNN, used to precompute sample
    explanations during training (artifacts/shap_precomputed.json).

    Returns (explainer, background) so the caller can persist the background
    sample to artifacts/shap_background.pkl.
    """
    import shap

    background = X_train_smote[np.random.choice(len(X_train_smote), background_size, replace=False)]

    def dnn_predict(data):
        return dnn_model.predict(data, verbose=0).flatten()

    print("Fitting SHAP KernelExplainer (this may take a moment)...")
    explainer = shap.KernelExplainer(dnn_predict, background)
    return explainer, background


def precompute_shap_samples(explainer, dnn_model, X_test_scaled, feature_cols, n_samples=20):
    """Precompute SHAP values for the first n_samples test instances."""
    def dnn_predict(data):
        return dnn_model.predict(data, verbose=0).flatten()

    sample_test = X_test_scaled[:n_samples]
    shap_values_sample = explainer.shap_values(sample_test, nsamples=100)

    shap_precomputed = {
        'feature_names': feature_cols,
        'base_value': float(explainer.expected_value),
        'samples': []
    }
    for i in range(len(sample_test)):
        shap_precomputed['samples'].append({
            'shap_values': [round(float(v), 6) for v in shap_values_sample[i]],
            'feature_values': [round(float(v), 4) for v in sample_test[i]],
            'prediction': round(float(dnn_predict(sample_test[i:i + 1])[0]), 4)
        })
    return shap_precomputed


# ===================== SHAP-TO-UI MAPPING ==========================
# Maps technical SHAP features (V1-V28, Log_Amount, Hour) to human-readable
# categories for non-technical users. Shared between training-time precompute
# and the live /api/predict endpoint in backend/predictor.py.

SHAP_TO_UI_MAP = {
    'Log_Amount': 'amount',
    'Hour': 'hour',
    'V1': 'behavioral_pattern', 'V2': 'behavioral_pattern',
    'V3': 'behavioral_pattern', 'V4': 'behavioral_pattern',
    'V5': 'behavioral_pattern', 'V6': 'behavioral_pattern',
    'V7': 'behavioral_pattern', 'V8': 'behavioral_pattern',
    'V9': 'behavioral_pattern', 'V10': 'behavioral_pattern',
    'V11': 'behavioral_pattern', 'V12': 'behavioral_pattern',
    'V13': 'behavioral_pattern', 'V14': 'behavioral_pattern',
    'V15': 'behavioral_pattern', 'V16': 'behavioral_pattern',
    'V17': 'behavioral_pattern', 'V18': 'behavioral_pattern',
    'V19': 'behavioral_pattern', 'V20': 'behavioral_pattern',
    'V21': 'behavioral_pattern', 'V22': 'behavioral_pattern',
    'V23': 'behavioral_pattern', 'V24': 'behavioral_pattern',
    'V25': 'behavioral_pattern', 'V26': 'behavioral_pattern',
    'V27': 'behavioral_pattern', 'V28': 'behavioral_pattern',
}

UI_CATEGORY_LABELS = {
    'amount': 'Transaction Amount',
    'hour': 'Time of Day',
    'behavioral_pattern': 'Behavioral Pattern (PCA)'
}


def summarize_shap_to_ui(shap_contributions):
    """Aggregate SHAP values by UI category for human-readable summary."""
    category_totals = {}
    for c in shap_contributions:
        feat = c['feature']
        ui_cat = SHAP_TO_UI_MAP.get(feat, 'behavioral_pattern')
        if ui_cat not in category_totals:
            category_totals[ui_cat] = 0.0
        category_totals[ui_cat] += abs(c['shap_value'])

    summary = []
    for cat, total in sorted(category_totals.items(), key=lambda x: -x[1]):
        summary.append({
            'category': cat,
            'label': UI_CATEGORY_LABELS.get(cat, cat),
            'total_impact': round(total, 6)
        })
    return summary
