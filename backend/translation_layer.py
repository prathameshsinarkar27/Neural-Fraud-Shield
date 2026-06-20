"""
backend/translation_layer.py
==============================================================================
Translation Layer: maps human-readable UI parameters (amount, hour, location,
device status, transaction type, merchant history) to the 30-dimensional
feature vector (V1-V28, Log_Amount, Hour) that the DNN/XGBoost models expect.
==============================================================================
"""

import numpy as np


def translate_ui_to_features(params, feature_cols, feature_stats):
    """
    Maps custom UI parameters to a 30-dimensional feature vector.
    Returns (features, ui_contributions) where ui_contributions tracks
    each UI parameter's risk contribution.
    """
    amount = float(params.get('amount', 50))
    hour = int(params.get('hour', 12))
    location = params.get('location', 'domestic')
    device_status = params.get('device_status', 'verified')
    transaction_type = params.get('transaction_type', 'in_store')
    merchant_history = params.get('merchant_history', 'established')

    # Start with overall means (neutral baseline)
    features = np.zeros(30)
    for i, col in enumerate(feature_cols):
        features[i] = feature_stats[col]['overall_mean']

    # Set Log_Amount and Hour directly
    features[28] = np.log1p(amount)  # Log_Amount
    features[29] = float(hour)       # Hour

    # Track individual UI contributions
    ui_contributions = {}

    # Location risk
    location_risk = {'domestic': 0.0, 'foreign': 0.4, 'high_risk_country': 0.8}
    loc_risk = location_risk.get(location, 0.2)
    ui_contributions['location'] = {
        'value': location,
        'risk_score': round(loc_risk, 4),
        'label': location.replace('_', ' ').title()
    }

    # Device risk
    device_risk = {'verified': 0.0, 'new_device': 0.3, 'compromised': 0.9}
    dev_risk = device_risk.get(device_status, 0.15)
    ui_contributions['device_status'] = {
        'value': device_status,
        'risk_score': round(dev_risk, 4),
        'label': device_status.replace('_', ' ').title()
    }

    # Transaction type risk
    txn_risk = {'in_store': 0.0, 'online': 0.2, 'atm': 0.3, 'wire_transfer': 0.6}
    tx_risk = txn_risk.get(transaction_type, 0.1)
    ui_contributions['transaction_type'] = {
        'value': transaction_type,
        'risk_score': round(tx_risk, 4),
        'label': transaction_type.replace('_', ' ').title()
    }

    # Merchant history risk
    merchant_risk = {'established': 0.0, 'new': 0.3, 'flagged': 0.7}
    mer_risk = merchant_risk.get(merchant_history, 0.15)
    ui_contributions['merchant_history'] = {
        'value': merchant_history,
        'risk_score': round(mer_risk, 4),
        'label': merchant_history.replace('_', ' ').title()
    }

    # Amount risk
    amount_risk = 0.0
    if amount > 5000:
        amount_risk = 0.5
    elif amount > 2000:
        amount_risk = 0.3
    elif amount > 500:
        amount_risk = 0.1
    elif amount < 1:
        amount_risk = 0.2
    ui_contributions['amount'] = {
        'value': amount,
        'risk_score': round(amount_risk, 4),
        'label': f'${amount:,.2f}'
    }

    # Hour risk
    hour_risk = 0.0
    if hour >= 23 or hour <= 4:
        hour_risk = 0.3
    elif hour >= 21 or hour <= 6:
        hour_risk = 0.1
    ui_contributions['hour'] = {
        'value': hour,
        'risk_score': round(hour_risk, 4),
        'label': f'{hour}:00'
    }

    # Compute overall risk score
    risk_score = loc_risk + dev_risk + tx_risk + mer_risk + amount_risk + hour_risk
    risk_score = min(risk_score / 3.0, 1.0)

    # Interpolate PCA features between legit and fraud distributions
    for i, col in enumerate(feature_cols[:28]):  # V1-V28
        corr = abs(feature_stats[col]['correlation_with_fraud'])
        mean_l = feature_stats[col]['mean_legit']
        mean_f = feature_stats[col]['mean_fraud']
        std_l = feature_stats[col]['std_legit']

        weight = risk_score * (0.3 + 0.7 * corr)
        features[i] = mean_l * (1 - weight) + mean_f * weight
        features[i] += np.random.normal(0, std_l * 0.1)

    return features, ui_contributions
