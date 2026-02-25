## Fraud Detection Project

This repository implements a **credit card fraud detection** system using a trained machine‑learning model on the popular Kaggle `creditcard.csv` dataset.

### Overview

- **Dataset**: `creditcard.csv` with features `Time`, `V1`–`V28`, `Amount` and target `Class` (`0` = normal, `1` = fraud).
- **Model artifacts** (kept locally, **ignored by Git**): `best_fraud_model.pkl`, `fraud_scaler.pkl`, `feature_names.pkl`, `all_fraud_models.pkl`, `rf_model.pkl`.
- **Code**:
  - `fraud_metrics.py` / `get_metrics.py` – evaluation and reporting
  - `predict_fraud.py` – demo, interactive, and batch predictions
  - `conversational_fraud_detector.py` – conversational fraud analysis using LangGraph + LLMs

### Setup

- **Python**: 3.10+ recommended.
- Create and activate a virtual environment:

```bash
cd Fraud_Detection
python -m venv venv
source venv/bin/activate           # macOS / Linux
# venv\Scripts\activate            # Windows
```

- Install required packages (example minimal set):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib python-dotenv \
            langgraph langchain-core langchain-community langchain-openai
```

### Data

- Place `creditcard.csv` in the project root.
- Required columns:
  - `Time`, `V1`–`V28`, `Amount`
  - `Class` (`0` = normal, `1` = fraud) for evaluation.

### Model artifacts

- Trained model and related files are expected in the project root:
  - `best_fraud_model.pkl` – main model used by the scripts
  - `fraud_scaler.pkl` – scaler fitted during training
  - `feature_names.pkl` – list of feature names
  - `all_fraud_models.pkl`, `rf_model.pkl` – extra models for analysis
- These `.pkl` files are **not committed to Git** (see `.gitignore`).  
  You should generate or copy them locally before running the scripts.

### Evaluating the model

`get_metrics.py` uses `FraudDetectionMetrics` from `fraud_metrics.py` to evaluate the trained model and create reports/plots.

From the project root:

```bash
python get_metrics.py                # uses creditcard.csv
# or
python get_metrics.py path/to/your_dataset.csv
```

Outputs:

- `fraud_metrics_report.txt` – human‑readable text report
- `fraud_metrics_report.json` – metrics as JSON
- `metrics_plots/` – ROC, PR curve, confusion matrix, feature importance and other plots

### Predictions

All prediction utilities live in `predict_fraud.py`.

- **Demo on sample normal + fraud transactions**:

```bash
python predict_fraud.py
```

- **Interactive single transaction**:

```bash
python -c "import predict_fraud as pf; pf.interactive_prediction()"
```

- **Batch CSV prediction** (e.g. on `creditcard.csv`):

```bash
python -c "import predict_fraud as pf; pf.batch_prediction('creditcard.csv')"
```

Batch output is written to `fraud_predictions.csv` with fraud probability, predicted label, and risk level for each row.

### Conversational fraud detector (optional)

`conversational_fraud_detector.py` wraps the model in a LangGraph workflow and uses an LLM to:

- Understand the user’s intent (analyze a transaction vs. general question).
- Extract transaction fields from free text.
- Run the ML model and explain the result in natural language.

Supported LLM backends (in order):

1. OpenAI ChatGPT (`OPENAI_API_KEY`)
2. DashScope Qwen (`DASHSCOPE_API_KEY`)
3. Local Ollama Qwen2 (`qwen2:7b` at `http://localhost:11434`)

You can control which backend is used by setting the corresponding environment variable (or `.env` file).

### Notes

- The dataset is **highly imbalanced** (fraud ~0.17%), so pay attention to precision, recall, PR curves, and confusion matrix, not just accuracy.
- If you retrain the model, regenerate all `.pkl` artifacts together so that model, scaler, and `feature_names` stay in sync.

