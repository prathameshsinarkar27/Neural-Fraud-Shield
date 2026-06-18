# Neural Fraud Shield

Production-grade modular refactor of the Financial Fraud Detection project.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train models and generate artifacts
python train.py

# Start the API server
python app.py
```

## Project Structure

```
neural-fraud-shield/
├── train.py                    # Training entry point
├── app.py                      # Flask API entry point
├── config/settings.py          # Central configuration
├── data/                       # Data loading & preprocessing
├── models/                     # DNN + XGBoost architectures & trainer
├── evaluation/                 # Metrics, plots, explainability
├── api/                        # Routes, predictor, artifact loader
├── services/                   # Translation layer, SHAP/LIME services
├── frontend/                   # HTML, CSS, JS
├── utils/                      # Helpers and constants
├── tests/                      # Test suite
└── artifacts/                  # Trained models and JSON artifacts
```

> Documentation updated in Phase 18.
