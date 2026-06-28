# 🛡️ Neural Fraud Shield — DNN Financial Fraud Detection System

> Financial Fraud Detection using Deep Neural Networks with Explainable AI

An industry-level fraud detection system combining a Deep Neural Network (DNN) with XGBoost baseline comparison, SHAP/LIME explainability, and a real-time dark-mode dashboard.

🔗 **Live Demo:** [neural-fraud-shield.onrender.com](https://neural-fraud-shield.onrender.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Model Details](#model-details)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [API Endpoints](#api-endpoints)
- [Dashboard Sections](#dashboard-sections)
- [Dataset](#dataset)
- [Results](#results)

---

## 🔍 Overview

This project addresses the critical industry need for **banking risk management** by building a production-ready fraud detection pipeline. It uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds) and provides:

- A **Deep Neural Network** trained with Adam Optimizer & Backpropagation
- **XGBoost** as a strong baseline for model comparison
- **SHAP & LIME** for explainable predictions
- A **translation layer** mapping human-readable inputs to PCA feature vectors
- A **real-time dashboard** with 7 interactive sections

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│             Frontend (frontend/index.html)               │
│   Chart.js · Dark Mode UI · 7 Dashboard Sections         │
└─────────────────────┬───────────────────────────────────┘
                       │ REST API (fetch)
┌─────────────────────▼───────────────────────────────────┐
│                  Flask API (app.py)                       │
│  Routes → backend/predictor.py → backend/translation_layer│
│  DNN Inference · XGBoost Inference · SHAP · LIME          │
└─────────────────────┬───────────────────────────────────┘
                       │ Loads from disk
┌─────────────────────▼───────────────────────────────────┐
│              Training Pipeline (train.py)                 │
│  data/ (load, engineer, scale, SMOTE)                     │
│  models/ (DNN, XGBoost, metrics, SHAP fitting)            │
└─────────────────────────────────────────────────────────┘
```

The codebase is organized so each layer has one job: `data/` turns the raw CSV into model-ready features, `models/` defines and evaluates the two models, `backend/` serves live predictions, and `frontend/` is the dashboard. `train.py` and `app.py` are the only two entry points.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **DNN Model** | 4-layer deep neural network (128→64→32→16) with BatchNorm, ReLU, Dropout |
| **XGBoost Baseline** | Gradient-boosted trees for comparison, proving DNN superiority |
| **SMOTE** | Handles extreme class imbalance (0.17% fraud) |
| **SHAP** | TreeExplainer (XGBoost) for live inference; KernelExplainer (DNN) for training-time precomputed samples |
| **LIME** | Local interpretable model-agnostic explanations |
| **Translation Layer** | Maps UI inputs (location, device, etc.) → 30-dim PCA vector |
| **UI Risk Factors** | Top 3 human-readable risk contributors per prediction |
| **SHAP-to-UI Mapping** | Aggregates technical SHAP values into understandable categories |
| **Live Feed** | Real-time transaction simulation scored every 1.5s |
| **Interactive Threshold** | Slider (0.00–1.00) recalculating Precision/Recall/FPR |

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **TensorFlow/Keras** — DNN architecture & training
- **XGBoost** — Baseline model
- **scikit-learn** — Preprocessing, metrics, t-SNE
- **imbalanced-learn** — SMOTE oversampling
- **SHAP** — SHapley Additive exPlanations
- **LIME** — Local Interpretable Model-agnostic Explanations
- **Flask + Flask-CORS** — REST API server

### Frontend
- **HTML5/CSS3/JavaScript** — Single-page application
- **Chart.js 4** — All visualizations (bar, line, scatter, doughnut)
- **Font Awesome 6** — Icons
- **Inter + JetBrains Mono** — Typography

---

## 📁 Project Structure

```
Neural-Fraud-Shield/
├── train.py                    # Training orchestrator (run first)
├── app.py                      # Flask API server (run second)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── creditcard.csv          # Dataset (auto-downloaded or manual — gitignored)
│   ├── preprocessing.py        # Dataset loading, train/test split, scaling, SMOTE
│   └── feature_engineering.py  # Log_Amount/Hour engineering, feature stats & importance
│
├── models/
│   ├── dnn_model.py            # DNN architecture, compile, training loop
│   ├── xgb_model.py            # XGBoost baseline build/train/predict
│   └── explainability.py       # Metrics, ROC/PR/threshold curves, SHAP fitting,
│                                # and SHAP-to-UI category mapping (shared with backend/)
│
├── backend/
│   ├── predictor.py            # Loads artifacts, runs DNN+XGBoost+SHAP+LIME inference
│   └── translation_layer.py    # Maps UI params → 30-dim PCA feature vector
│
├── frontend/
│   ├── index.html              # Dashboard markup
│   ├── styles.css              # Dark-mode dashboard styling
│   └── app.js                  # Chart.js wiring, API calls, live feed, all UI logic
│
├── artifacts/                  # Generated by train.py
│   ├── dnn_model.h5
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   ├── training_history.json
│   ├── feature_stats.json
│   ├── feature_cols.json
│   ├── feature_importance.json
│   ├── tsne_data.json
│   ├── distribution_data.json
│   ├── hourly_data.json
│   ├── amount_distribution.json
│   ├── smote_data.json
│   ├── shap_precomputed.json
│   ├── shap_background.pkl
│   └── lime_training_data.npy

```

---

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Dataset

Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`. If it is missing, `train.py` will attempt to auto-download it via `kagglehub`.

### 3. Train the Model

```bash
python train.py
```

This will:
- Load `data/creditcard.csv` (or auto-download it)
- Apply SMOTE to balance classes
- Train the DNN with Adam Optimizer & Backpropagation
- Train the XGBoost baseline
- Compute all metrics, t-SNE, SHAP/LIME explainers
- Save all artifacts to `./artifacts/`

> ⏱️ Training takes approximately 5–15 minutes depending on hardware.

### 4. Start the Server

```bash
python app.py
```

### 5. Open the Dashboard

Navigate to **http://localhost:5000** in your browser. The status indicator in the header will turn green once the backend is connected and the models are active.

---

## 🧠 Model Details

### Deep Neural Network (DNN)

| Parameter | Value |
|-----------|-------|
| Architecture | Input(30) → Dense(128) → BN → ReLU → Dropout(0.3) → Dense(64) → BN → ReLU → Dropout(0.4) → Dense(32) → BN → ReLU → Dropout(0.3) → Dense(16) → ReLU → Dense(1, Sigmoid) |
| Optimizer | **Adam** (lr=0.001, β₁=0.9, β₂=0.999, ε=1e-7) |
| Loss Function | Binary Cross-Entropy |
| Training Method | **Backpropagation** (via Keras `.fit()`) |
| Batch Size | 256 |
| Early Stopping | patience=5, monitoring val_auc |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

Defined in `models/dnn_model.py`.

### XGBoost Baseline

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

Defined in `models/xgb_model.py`.

---

## 🔬 Explainable AI (XAI)

All explainability code lives in `models/explainability.py` (shared logic, metrics, SHAP fitting, SHAP-to-UI mapping) and `backend/predictor.py` (live inference).

### SHAP (SHapley Additive exPlanations)
- **Training time:** `KernelExplainer` against the DNN with 100 background samples, precomputed for sample test instances
- **Serving time:** `TreeExplainer` against XGBoost, computed live for every `/api/predict` request
- Identifies the **top contributing feature** (highest |SHAP value|)
- Aggregates into human-readable categories via the **SHAP-to-UI mapping**:
  - `Log_Amount` → "Transaction Amount"
  - `Hour` → "Time of Day"
  - `V1–V28` → "Behavioral Pattern (PCA)"

### LIME (Local Interpretable Model-agnostic Explanations)
- `LimeTabularExplainer` with fraud/legit class names
- Top 10 feature contributions per prediction
- Provides complementary local explanations to SHAP

### Translation Layer
`backend/translation_layer.py` maps 6 human-readable parameters to 30-dim PCA vectors:
- **Amount** → `Log_Amount` (direct)
- **Hour** → `Hour` (direct)
- **Location** (domestic/foreign/high-risk) → shifts V1, V3, V14
- **Device Status** (verified/new/compromised) → shifts V4, V11, V16
- **Transaction Type** (in-store/online/ATM/wire) → shifts V7, V12, V17
- **Merchant History** (established/new/flagged) → shifts V10, V14

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend dashboard |
| `GET` | `/api/metrics` | All model metrics (DNN + XGBoost) |
| `GET` | `/api/training_history` | Training curves + architecture |
| `GET` | `/api/feature_importance` | Feature correlation rankings |
| `GET` | `/api/tsne` | t-SNE 2D visualization data |
| `GET` | `/api/distribution` | Feature distribution comparison |
| `GET` | `/api/hourly` | Hourly transaction volumes |
| `GET` | `/api/amount_distribution` | Amount bucket analysis |
| `GET` | `/api/smote` | SMOTE before/after data |
| `GET` | `/api/shap_precomputed` | Pre-computed SHAP values |
| `POST` | `/api/predict` | Run DNN + XGBoost inference with SHAP/LIME |
| `GET` | `/api/live_transaction` | Random scored transaction |
| `GET` | `/api/threshold_analysis` | Threshold sweep data |

### POST `/api/predict` — Request Body

```json
{
  "amount": 1200,
  "hour": 2,
  "location": "foreign",
  "transaction_type": "atm",
  "merchant_history": "new",
  "device_status": "new_device"
}
```

### POST `/api/predict` — Response

```json
{
  "fraud_probability": 0.7823,
  "xgb_probability": 0.6541,
  "confidence": 0.8912,
  "decision": "FRAUD",
  "top_feature": {
    "feature": "V14",
    "shap_value": -0.1234,
    "direction": "increases fraud risk",
    "ui_category": "Behavioral Pattern (PCA)"
  },
  "top_ui_factors": [
    { "factor": "Device Status", "value": "New Device", "risk_score": 0.3, "risk_level": "Medium" }
  ],
  "shap_summary": [
    { "category": "behavioral_pattern", "label": "Behavioral Pattern (PCA)", "total_impact": 0.4521 }
  ],
  "shap": { "base_value": 0.5, "contributions": ["..."] },
  "lime": { "contributions": ["..."] },
  "recommended_action": "Block transaction immediately."
}
```

---

## 📊 Dashboard Sections

1. **System Dashboard** — KPI cards, hourly volume, detection rates, class distribution, risk breakdown
2. **Live Transaction Feed** — Real-time scoring every 1.5s with Pause/Resume
3. **Transaction Analyzer** — 6 quick scenarios, custom form, SHAP/LIME visualizations, top risk factors
4. **Feature Analysis** — Pearson correlation, t-SNE scatter, distribution comparison
5. **Model Performance** — DNN vs XGBoost showdown, ROC/PR curves, confusion matrix, threshold slider
6. **Training Visualization** — Backpropagation flow diagram, learning curves with best epoch, Adam optimizer params, architecture diagram, weight evolution
7. **Dataset Insights** — Class imbalance (log scale), SMOTE before/after, feature dictionary

---

## 📈 Dataset

**Kaggle Credit Card Fraud Detection Dataset**

| Property | Value |
|----------|-------|
| Total Transactions | 284,807 |
| Fraudulent | 492 (0.172%) |
| Features | V1–V28 (PCA), Amount, Time |
| Engineered | Log_Amount, Hour |
| Model Input | 30 features |

---

## 🏆 Results

| Metric | DNN | XGBoost |
|--------|-----|---------|
| ROC-AUC | **~0.98+** | ~0.97+ |
| PR-AUC | **~0.80+** | ~0.75+ |
| Recall | **~0.82+** | ~0.80+ |
| F1 Score | **~0.85+** | ~0.82+ |

> *Exact values depend on the training run. The DNN consistently outperforms XGBoost on complex non-linear fraud patterns.*

---

## 📝 Key Technical Highlights

1. **Adam Optimizer** — Adaptive learning rates using 1st & 2nd moment estimates of the gradient
2. **Backpropagation** — Chain rule computes ∂Loss/∂weights for every layer; visualized interactively in the Training tab
3. **SMOTE** — Synthetic Minority Oversampling addresses the extreme 0.17% fraud-class imbalance
4. **SHAP + LIME** — Dual XAI approach providing both global feature attribution and local per-prediction explanations
5. **Translation Layer** — Bridges human-readable UI inputs to the PCA feature space the models were trained on
6. **DNN > XGBoost** — Deep learning captures complex non-linear patterns in PCA space that gradient-boosted trees cannot

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

