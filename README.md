# Production Fraud Detection System

A production-grade machine learning system for detecting fraudulent transactions, built with a focus on handling real-world constraints like extreme class imbalance, threshold tuning, and robust API design.

## 🎯 Project Overview

Fraud detection in the real world is challenging because fraudulent transactions are extremely rare compared to legitimate ones (often < 1%). This project demonstrates how to build a robust system that addresses these challenges head-on.

### Key Features

1. **Imbalanced Data Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) combined with XGBoost's native `scale_pos_weight` to effectively learn from highly imbalanced data (1:65 ratio).
2. **Business-Driven Threshold Tuning**: Rejects the default 0.5 probability threshold. Instead, it tunes the decision threshold based on the F1-score to optimize the trade-off between catching fraud (Recall) and minimizing false alarms (Precision).
3. **Robust Evaluation**: Evaluates the model using Precision, Recall, F1-Score, and Precision-Recall AUC, rather than misleading metrics like Accuracy. Includes a detailed business impact analysis.
4. **Production-Ready API**: A FastAPI-based REST endpoint that handles edge cases, missing values (via median imputation), and invalid inputs with detailed error messages.

## 📊 Model Evaluation & Trade-offs

When dealing with fraud, **Accuracy is a misleading metric**. If 99% of transactions are legitimate, a model that simply predicts "Not Fraud" every time will be 99% accurate, but completely useless.

Instead, we focus on:
- **Precision**: When the model flags a transaction as fraud, how often is it correct? (Minimizing false positives/customer friction)
- **Recall**: Of all actual fraudulent transactions, how many did we catch? (Minimizing financial loss)
- **F1-Score**: The harmonic mean of Precision and Recall.

### Threshold Tuning Impact

By default, classification models use a 0.5 threshold. In our imbalanced scenario, this leads to too many false positives. By tuning the threshold to **0.90**, we achieved:

| Metric | Default Threshold (0.50) | Tuned Threshold (0.90) | Impact |
|--------|--------------------------|------------------------|--------|
| **Precision** | 0.5225 | 0.6927 | **+17.02%** (Fewer false alarms) |
| **Recall** | 0.9667 | 0.9467 | -2.00% (Slight drop in detection) |
| **F1-Score** | 0.6784 | 0.8000 | **+12.16%** (Better overall balance) |

*Trade-off*: We sacrificed a tiny amount of recall (2%) to gain a massive improvement in precision (17%). This means significantly fewer legitimate customers will have their cards blocked, while we still catch ~95% of all fraud.

### Business Impact Analysis

Based on our test set (20,000 transactions, 300 fraudulent):
- **Actual Fraud Amount**: $75,000
- **Loss without Model**: $71,250 (assuming 95% loss rate)
- **Loss with Model**: $4,430 (missed fraud + cost of investigating false positives)
- **Estimated Savings**: $66,820
- **ROI**: 93.8%

## 🛠️ Project Structure

```
fraud-detection-system/
├── api/
│   └── app.py              # FastAPI application
├── data/
│   ├── generate_data.py    # Script to generate realistic imbalanced data
│   ├── transactions.csv    # Raw generated dataset
│   ├── train.csv           # Training split
│   └── test.csv            # Testing split
├── models/
│   ├── train.py            # Model training and threshold tuning
│   ├── evaluate.py         # Comprehensive evaluation script
│   ├── model.pkl           # Trained XGBoost model
│   ├── scaler.pkl          # Feature scaler
│   └── imputer.pkl         # Missing value imputer
├── notebooks/
│   └── evaluation_metrics.png # Visualizations of model performance
├── tests/
│   └── test_api.py         # Comprehensive API test suite
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fraud-detection-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

1. **Generate Data**:
   ```bash
   python data/generate_data.py
   ```

2. **Train Model**:
   ```bash
   python models/train.py
   ```

3. **Evaluate Model**:
   ```bash
   python models/evaluate.py
   ```

4. **Start API Server**:
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```

5. **Run Tests**:
   ```bash
   pytest tests/test_api.py -v
   ```

## 🔌 API Usage

### Single Prediction

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "transaction_amount": 150.50,
  "merchant_risk_score": 0.15,
  "user_age": 35,
  "days_since_account_creation": 500,
  "transaction_hour": 14,
  "num_transactions_today": 3,
  "device_trust_score": 0.85,
  "failed_login_attempts": 0,
  "international_transaction": 0,
  "card_present": 1
}
```

**Response**:
```json
{
  "fraud_probability": 0.0012,
  "is_fraud": false,
  "decision_threshold": 0.90,
  "confidence": "high",
  "timestamp": "2024-04-21T10:30:45Z"
}
```

### Edge Case Handling

The API is designed to be robust against real-world messy data:
- **Missing Values**: Handled gracefully via median imputation. You can omit fields or send `null`.
- **Invalid Inputs**: Strict validation ensures values are within logical bounds (e.g., age between 18-100, binary fields are 0 or 1). Returns detailed 422 errors.
- **All Missing**: Rejects requests where every single feature is missing with a 400 error.

## 📝 License

This project is licensed under the MIT License.
