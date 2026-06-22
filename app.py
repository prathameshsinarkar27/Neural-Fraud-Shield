"""
==============================================================================
NNARL Project 1: Financial Fraud Detection - Flask API Server
app.py
==============================================================================
This server:
1. Loads trained DNN model and XGBoost baseline 
2. Serves the frontend dashboard (frontend/index.html, styles.css, app.js)
3. Exposes REST endpoints that serve precomputed artifacts and live
   inference (with SHAP/LIME explanations) to the dashboard
==============================================================================
"""

import os
import json
import warnings

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from backend.predictor import FraudPredictor

warnings.filterwarnings('ignore')

ARTIFACTS_DIR = "artifacts"

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)


# ===================== LOAD ARTIFACTS ======================================

print("Loading dashboard artifacts...")

with open(f"{ARTIFACTS_DIR}/metrics.json") as f:
    metrics_data = json.load(f)
with open(f"{ARTIFACTS_DIR}/training_history.json") as f:
    training_history = json.load(f)
with open(f"{ARTIFACTS_DIR}/feature_importance.json") as f:
    feature_importance = json.load(f)
with open(f"{ARTIFACTS_DIR}/tsne_data.json") as f:
    tsne_data = json.load(f)
with open(f"{ARTIFACTS_DIR}/distribution_data.json") as f:
    distribution_data = json.load(f)
with open(f"{ARTIFACTS_DIR}/hourly_data.json") as f:
    hourly_data = json.load(f)
with open(f"{ARTIFACTS_DIR}/amount_distribution.json") as f:
    amount_distribution = json.load(f)
with open(f"{ARTIFACTS_DIR}/smote_data.json") as f:
    smote_data = json.load(f)
with open(f"{ARTIFACTS_DIR}/shap_precomputed.json") as f:
    shap_precomputed = json.load(f)

predictor = FraudPredictor(artifacts_dir=ARTIFACTS_DIR)

print("All artifacts loaded successfully!")

# ===================== API ENDPOINTS =======================================


@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics_data)


@app.route('/api/training_history', methods=['GET'])
def get_training_history():
    return jsonify(training_history)


@app.route('/api/feature_importance', methods=['GET'])
def get_feature_importance():
    return jsonify(feature_importance)


@app.route('/api/tsne', methods=['GET'])
def get_tsne():
    return jsonify(tsne_data)


@app.route('/api/distribution', methods=['GET'])
def get_distribution():
    return jsonify(distribution_data)


@app.route('/api/hourly', methods=['GET'])
def get_hourly():
    return jsonify(hourly_data)


@app.route('/api/amount_distribution', methods=['GET'])
def get_amount_distribution():
    return jsonify(amount_distribution)


@app.route('/api/smote', methods=['GET'])
def get_smote():
    return jsonify(smote_data)


@app.route('/api/shap_precomputed', methods=['GET'])
def get_shap_precomputed():
    return jsonify(shap_precomputed)