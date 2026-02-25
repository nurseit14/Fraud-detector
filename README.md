## Fraud Detection Project

This repository contains a **credit card fraud detection** pipeline built around a trained machine‑learning model and rich evaluation tooling. It includes:

- **Model artifacts**: `best_fraud_model.pkl`, `fraud_scaler.pkl`, `feature_names.pkl`, `all_fraud_models.pkl`, `rf_model.pkl`
- **Evaluation & reporting**: `fraud_metrics.py`, `get_metrics.py`, `fraud_metrics_report.txt`, `fraud_metrics_report.json`, `metrics_plots/`
- **Prediction tools**: `predict_fraud.py` for demo, interactive, and batch predictions
- **Conversational interface**: `conversational_fraud_detector.py` and LangGraph workflows for natural‑language fraud analysis
- **Dataset**: `creditcard.csv` (Kaggle credit card fraud dataset)

### 1. Environment setup

- **Python version**: Python 3.10+ is recommended.
- A local virtual environment is already present in `venv/` (created by PyCharm). You can either use it directly from the IDE or create a fresh one:

```bash
python -m venv venv
source venv/bin/activate           # macOS / Linux
# venv\Scripts\activate            # Windows (PowerShell or cmd)

pip install -r requirements.txt    # if you have one
```

If there is no `requirements.txt`, install at least:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `python-dotenv`
- `langgraph`
- `langchain-core`
- `langchain-community`
- `langchain-openai`

### 2. Data

- The project expects the **Kaggle credit card fraud dataset** as `creditcard.csv` in the project root.
- The CSV must contain:
  - Features `Time`, `V1`–`V28`, `Amount`
  - Target column `Class` (`0` = normal, `1` = fraud)

### 3. Model artifacts

The repository already includes trained artifacts:

- `best_fraud_model.pkl` – main fraud detection model
- `fraud_scaler.pkl` – feature scaler used during training
- `feature_names.pkl` – ordered list of feature names
- `all_fraud_models.pkl` – collection of different models (used for feature importance)
- `rf_model.pkl` – stored Random Forest model

These are loaded by the scripts and should remain in the project root.

### 4. Evaluating the model & generating metrics

**Core evaluation class**  
`fraud_metrics.py` defines `FraudDetectionMetrics`, which:

- Loads the trained model, scaler, and feature names
- Evaluates metrics (accuracy, precision, recall, F1, ROC‑AUC, average precision, specificity, sensitivity, FPR/FNR)
- Computes ROC and Precision‑Recall curves and a full classification report
- Can run cross‑validation and generate a human‑readable report

**Command‑line entrypoint**  
`get_metrics.py` is the main CLI script to evaluate the model on a dataset and generate reports and plots.

Run (from the project root):

```bash
python get_metrics.py                # uses creditcard.csv by default
# or
python get_metrics.py path/to/your_dataset.csv
```

This will:

- Load `best_fraud_model.pkl`, `fraud_scaler.pkl`, `feature_names.pkl`
- Evaluate performance on the provided dataset
- Generate:
  - `fraud_metrics_report.txt` – readable text report (also printed to console)
  - `fraud_metrics_report.json` – JSON with all metrics
  - A set of plots under `metrics_plots/`, including:
    - `roc_curve.png`
    - `precision_recall_curve.png`
    - `confusion_matrix.png`
    - `feature_importance.png`
    - `metrics_comparison.png`
    - `confusion_matrix_pie.png`

### 5. Making predictions

`predict_fraud.py` provides multiple ways to use the trained model.

#### 5.1 Demo on sample transactions

Runs predictions on one normal and one fraud transaction sampled from `creditcard.csv`:

```bash
python predict_fraud.py
```

This prints for each transaction:

- Fraud probability
- Risk level (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
- Recommendation (`APPROVE`, `REVIEW`, `INVESTIGATE`, `BLOCK`)
- A brief feature importance analysis for the fraud example

#### 5.2 Interactive single‑transaction prediction

Interactively type in `Amount`, `Time`, and `V1`–`V28`:

```bash
python -c "import predict_fraud as pf; pf.interactive_prediction()"
```

You will be prompted for numeric values (press Enter to accept defaults of `0` or `100` etc.). The script then prints:

- Fraud probability
- Boolean fraud flag
- Risk level
- Confidence
- Recommendation
- Feature importance overview

#### 5.3 Batch prediction from CSV

Run predictions for all rows in a CSV file:

```bash
python -c "import predict_fraud as pf; pf.batch_prediction('creditcard.csv')"
```

Input requirements:

- Columns `Time`, `V1`–`V28`, `Amount`
- Optional `Class` column for ground‑truth labels (if present, accuracy columns are added)

Output:

- `fraud_predictions.csv` in the project root containing:
  - `Transaction_ID`
  - `Fraud_Probability`
  - `Predicted_Fraud`
  - `Risk_Level`
  - Optional `Actual_Fraud` and `Correct_Prediction` if `Class` was provided

### 6. Conversational fraud detection (LangGraph + LLM)

`conversational_fraud_detector.py` implements a **conversational interface** on top of the trained model using LangGraph and various chat LLM backends.

#### 6.1 LLM backends

The class `ConversationalFraudDetector` tries LLMs in this order:

1. **OpenAI ChatGPT** (`gpt-3.5-turbo`) if `OPENAI_API_KEY` is set
2. **DashScope Qwen** (`qwen-turbo`) if `DASHSCOPE_API_KEY` is set
3. **Local Ollama Qwen2** (`qwen2:7b`) via `http://localhost:11434` as a fallback

Set your environment variables (or use a `.env` file):

```bash
export OPENAI_API_KEY="sk-..."              # or
export DASHSCOPE_API_KEY="your-dashscope-key"
```

For the Ollama fallback, ensure:

- Ollama is installed and running
- The `qwen2:7b` model is pulled:

```bash
ollama pull qwen2:7b
```

#### 6.2 What the workflow does

The LangGraph state (`ChatState`) contains:

- `messages` – full conversation history
- `current_transaction` – parsed transaction data (Amount, Time, V1–V28)
- `fraud_analysis` – outputs from the ML model (probability, is_fraud, confidence, top features)
- `conversation_context` – intent and routing metadata

The workflow nodes:

- **`understand_intent`** – classifies the user message as:
  - `analyze_transaction`
  - `general_query`
  - `end_conversation`
- **`extract_transaction`** – uses the LLM to extract structured transaction data from free‑text
- **`analyze_fraud`** – scales features, runs the ML model, and computes fraud probability and top features
- **`generate_response`** – turns the analysis into a natural‑language explanation
- **`handle_general_query`** – answers non‑transaction questions about fraud detection

You can import and use it in your own scripts, for example:

```python
from conversational_fraud_detector import ConversationalFraudDetector
from langchain_core.messages import HumanMessage

detector = ConversationalFraudDetector()
state = {
    "messages": [HumanMessage(content="Please check a $500 transaction at 10:30 with V1=-1.2, V2=0.5")],
    "current_transaction": {},
    "fraud_analysis": {},
    "conversation_context": {},
}

result_state = detector.workflow.invoke(state)
```

### 7. Files overview

- `creditcard.csv` – input dataset (features + `Class` label)
- `fraud_metrics.py` – metrics computation and reporting utilities
- `get_metrics.py` – command‑line script to evaluate the model and generate reports/plots
- `predict_fraud.py` – interactive, demo, and batch prediction utilities
- `conversational_fraud_detector.py` – LangGraph‑powered conversational fraud analysis using LLMs
- `fraud_metrics_report.txt` / `.json` – saved evaluation reports
- `metrics_plots/` – generated visualizations
- `*.pkl` files – trained model, scaler, feature names, and auxiliary models
- `venv/` – project virtual environment (can be recreated if needed)

### 8. Notes & best practices

- **Imbalanced data**: The dataset is highly imbalanced (fraud ~0.17%). Favor **recall** and **precision‑recall curves** when assessing performance, not only accuracy.
- **Threshold tuning**: Use methods like `FraudDetectionMetrics.find_optimal_threshold` to tune the decision threshold for your specific risk/operational constraints.
- **Reproducibility**: Keep the same `creditcard.csv` schema and do not modify the order or names of features stored in `feature_names.pkl` unless you retrain and regenerate all artifacts.

