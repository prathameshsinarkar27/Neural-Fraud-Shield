"""
backend/predictor.py
==============================================================================
Load all trained model artifacts (DNN, XGBoost, scaler, feature stats) and
expose the live inference logic used by the Flask routes in app.py:
  - predict(data)        -> powers POST /api/predict
  - live_transaction()   -> powers GET  /api/live_transaction
==============================================================================
"""

import json
import pickle
import time

import numpy as np
import tensorflow as tf
import lime
import lime.lime_tabular
import shap

from backend.translation_layer import translate_ui_to_features
from models.explainability import SHAP_TO_UI_MAP, UI_CATEGORY_LABELS, summarize_shap_to_ui


class FraudPredictor:
    """Owns all loaded model artifacts and serves live fraud predictions."""

    def __init__(self, artifacts_dir="artifacts"):
        print("Loading model artifacts...")

        self.model = tf.keras.models.load_model(f"{artifacts_dir}/dnn_model.h5")
        with open(f"{artifacts_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{artifacts_dir}/xgb_model.pkl", "rb") as f:
            self.xgb_model = pickle.load(f)
        with open(f"{artifacts_dir}/feature_stats.json") as f:
            self.feature_stats = json.load(f)
        with open(f"{artifacts_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)

        # LIME explainer training data
        self.X_train_smote = np.load(f"{artifacts_dir}/lime_training_data.npy")

        print("Setting up SHAP TreeExplainer for XGBoost...")
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)

        print("All artifacts loaded successfully!")

    def predict(self, data):
        """
        Main prediction logic. Accepts custom UI parameters, translates them
        to features, runs DNN + XGBoost inference, and generates SHAP/LIME
        explanations. Returns the same response shape as the original app.py.
        """
        # Translate UI parameters to feature vector (with contribution tracking)
        raw_features, ui_contributions = translate_ui_to_features(
            data, self.feature_cols, self.feature_stats
        )

        # Scale features
        scaled_features = self.scaler.transform(raw_features.reshape(1, -1))

        # DNN prediction
        dnn_prob = float(self.model.predict(scaled_features, verbose=0).flatten()[0])

        # XGBoost prediction
        xgb_prob = float(self.xgb_model.predict_proba(scaled_features)[0][1])

        # Determine risk level
        threshold = float(data.get('threshold', 0.5))
        if dnn_prob >= 0.7:
            decision = "FRAUD"
            action = "Block transaction immediately. Freeze account and notify cardholder."
        elif dnn_prob >= threshold:
            decision = "REVIEW"
            action = "Flag for manual review. Temporarily hold funds pending verification."
        else:
            decision = "SAFE"
            action = "Approve transaction. No action required."

        confidence = abs(dnn_prob - threshold) / max(threshold, 1 - threshold)
        confidence = min(confidence, 1.0)

        # SHAP explanation
        top_feature = None
        shap_contributions = []
        shap_base_value = 0.5
        try:
            shap_values = self.shap_explainer.shap_values(scaled_features)
            for i, col in enumerate(self.feature_cols):
                shap_contributions.append({
                    'feature': col,
                    'shap_value': round(float(shap_values[0][i]), 6),
                    'feature_value': round(float(scaled_features[0][i]), 4),
                    'abs_shap': round(abs(float(shap_values[0][i])), 6)
                })
            shap_contributions.sort(key=lambda x: x['abs_shap'], reverse=True)
            shap_base_value = float(self.shap_explainer.expected_value)

            # Extract TOP contributing feature (highest absolute SHAP value)
            if shap_contributions:
                top = shap_contributions[0]
                top_feature = {
                    'feature': top['feature'],
                    'shap_value': top['shap_value'],
                    'direction': 'increases fraud risk' if top['shap_value'] > 0 else 'decreases fraud risk',
                    'ui_category': UI_CATEGORY_LABELS.get(
                        SHAP_TO_UI_MAP.get(top['feature'], 'behavioral_pattern'),
                        'Behavioral Pattern'
                    )
                }
        except Exception as e:
            print(f"SHAP error: {e}")

        # SHAP-to-UI summary
        shap_summary = summarize_shap_to_ui(shap_contributions) if shap_contributions else []

        # LIME explanation
        lime_contributions = []
        try:
            def predict_fn(x):
                return self.xgb_model.predict_proba(x)

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train_smote,
                feature_names=self.feature_cols,
                class_names=['Legitimate', 'Fraud'],
                mode='classification'
            )

            exp = lime_explainer.explain_instance(
                scaled_features[0],
                predict_fn,
                num_features=10
            )

            for feat, weight in exp.as_list():
                lime_contributions.append({
                    'feature': feat,
                    'weight': round(float(weight), 6),
                    'abs_weight': round(abs(float(weight)), 6)
                })
            lime_contributions.sort(key=lambda x: x['abs_weight'], reverse=True)
        except Exception as e:
            print(f"LIME error: {e}")

        # Top 3 UI factors based on contribution
        sorted_ui = sorted(ui_contributions.items(), key=lambda x: -x[1]['risk_score'])
        top_ui_factors = []
        for key, val in sorted_ui[:3]:
            if val['risk_score'] > 0:
                top_ui_factors.append({
                    'factor': key.replace('_', ' ').title(),
                    'value': val['label'],
                    'risk_score': val['risk_score'],
                    'risk_level': 'High' if val['risk_score'] >= 0.5 else 'Medium' if val['risk_score'] >= 0.2 else 'Low'
                })

        return {
            'fraud_probability': round(dnn_prob, 4),
            'xgb_probability': round(xgb_prob, 4),
            'confidence': round(confidence, 4),
            'decision': decision,
            'threshold': threshold,
            'recommended_action': action,
            'top_feature': top_feature,
            'top_ui_factors': top_ui_factors,
            'shap_summary': shap_summary,
            'ui_contributions': ui_contributions,
            'shap': {
                'base_value': shap_base_value,
                'contributions': shap_contributions[:15]
            },
            'lime': {
                'contributions': lime_contributions[:15]
            },
            'input_params': data,
            'translated_features': {col: round(float(raw_features[i]), 4) for i, col in enumerate(self.feature_cols)}
        }

    def live_transaction(self):
        """Generate and score a random transaction for the live feed simulation."""
        amounts = [5, 12, 25, 47, 85, 120, 250, 500, 1200, 3000, 8000]
        locations = ['domestic', 'domestic', 'domestic', 'foreign', 'high_risk_country']
        devices = ['verified', 'verified', 'verified', 'new_device', 'compromised']
        txn_types = ['in_store', 'in_store', 'online', 'online', 'atm', 'wire_transfer']
        merchants = ['established', 'established', 'established', 'new', 'flagged']

        params = {
            'amount': float(np.random.choice(amounts)) * np.random.uniform(0.5, 2.0),
            'hour': int(np.random.randint(0, 24)),
            'location': np.random.choice(locations),
            'device_status': np.random.choice(devices),
            'transaction_type': np.random.choice(txn_types),
            'merchant_history': np.random.choice(merchants)
        }

        raw_features, _ = translate_ui_to_features(params, self.feature_cols, self.feature_stats)
        scaled_features = self.scaler.transform(raw_features.reshape(1, -1))
        prob = float(self.model.predict(scaled_features, verbose=0).flatten()[0])

        if prob >= 0.7:
            status = "FRAUD"
        elif prob >= 0.5:
            status = "REVIEW"
        else:
            status = "SAFE"

        return {
            'timestamp': time.time(),
            'amount': round(params['amount'], 2),
            'location': params['location'],
            'device_status': params['device_status'],
            'transaction_type': params['transaction_type'],
            'merchant_history': params['merchant_history'],
            'hour': params['hour'],
            'fraud_probability': round(prob, 4),
            'status': status
        }
