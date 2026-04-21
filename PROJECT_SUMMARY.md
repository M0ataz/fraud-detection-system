# Fraud Detection System - Project Summary

## What Was Built

A **production-grade fraud detection system** that demonstrates how to properly handle real-world ML challenges:

### 1. Data Generation (100,000 transactions)
- **1.5% fraud rate** (1,500 fraudulent vs 98,500 legitimate)
- Realistic patterns: fraudsters use different merchants, higher amounts, newer accounts
- Missing values: 5% missing age, 3% missing merchant score, 2% missing device trust
- Temporal patterns: fraud more likely at odd hours
- **File**: `data/generate_data.py`

### 2. Model Training
- **Algorithm**: XGBoost with SMOTE oversampling
- **Imbalance Handling**: 
  - SMOTE: Generates synthetic fraud examples (1:1 ratio during training)
  - scale_pos_weight: Native XGBoost parameter = 65.67 (ratio of legitimate to fraud)
- **Preprocessing**: Median imputation + StandardScaler
- **File**: `models/train.py`

### 3. Threshold Tuning (THE CRITICAL PART)
- Default threshold (0.5) → Too many false positives
- Tuned threshold (0.9) → Better precision/recall trade-off
- **Impact**:
  - Precision: 52.25% → 69.27% (+17%)
  - Recall: 96.67% → 94.67% (-2%)
  - F1: 67.84% → 80.00% (+12%)
- **File**: `models/train.py` (find_optimal_threshold method)

### 4. Evaluation
- **Metrics**: Precision, Recall, F1, ROC-AUC, PR-AUC
- **Business Impact**: $66,820 savings on $75,000 fraud (93.8% ROI)
- **Confusion Matrix Analysis**: 284 TP, 16 FN, 126 FP, 19,574 TN
- **File**: `models/evaluate.py`

### 5. Production API
- **Framework**: FastAPI
- **Endpoint**: POST /predict
- **Features**:
  - Input validation (age 18-100, scores 0-1, etc.)
  - Missing value handling (median imputation)
  - Batch predictions (up to 1000 at once)
  - Detailed error messages
  - Logging
- **File**: `api/app.py`

### 6. Comprehensive Tests
- **19 test cases** covering:
  - Valid predictions
  - Missing values
  - Invalid inputs (out of range, wrong types)
  - Edge cases (min/max values)
  - Batch predictions
- **All tests pass** ✓
- **File**: `tests/test_api.py`

## Key Learnings

### ❌ What NOT to Do
1. **Use Accuracy as main metric** - 99% accuracy means nothing if 99% of data is legitimate
2. **Ignore class imbalance** - Model will learn to predict "not fraud" for everything
3. **Use default 0.5 threshold** - Leads to too many false positives in imbalanced scenarios
4. **Assume clean data** - Real data has missing values, outliers, and noise

### ✅ What TO Do
1. **Use Precision, Recall, F1** - These metrics reflect real business impact
2. **Handle imbalance explicitly** - SMOTE + scale_pos_weight
3. **Tune threshold based on business metrics** - Not accuracy
4. **Handle missing values** - Imputation strategy matters
5. **Validate inputs** - Reject invalid data early

## File Structure

```
fraud-detection-system/
├── api/
│   └── app.py                    # FastAPI application (19 test cases pass)
├── data/
│   ├── generate_data.py          # Generate 100k transactions
│   ├── train.csv                 # 80k training samples
│   └── test.csv                  # 20k test samples
├── models/
│   ├── train.py                  # Training + threshold tuning
│   ├── evaluate.py               # Comprehensive evaluation
│   ├── model.pkl                 # Trained XGBoost
│   ├── scaler.pkl                # StandardScaler
│   └── imputer.pkl               # SimpleImputer
├── tests/
│   └── test_api.py               # 19 passing tests
├── notebooks/
│   └── evaluation_metrics.png    # Visualizations
├── README.md                     # Full documentation
├── requirements.txt              # Dependencies
└── .gitignore                    # Git ignore rules
```

## How to Use

### 1. Generate Data
```bash
python data/generate_data.py
```
Output: 100,000 transactions with 1.5% fraud rate

### 2. Train Model
```bash
python models/train.py
```
Output: Trained model + threshold tuning results

### 3. Evaluate
```bash
python models/evaluate.py
```
Output: Comprehensive metrics + visualizations

### 4. Run API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 5. Test
```bash
pytest tests/test_api.py -v
```
Output: 19/19 tests passing

## API Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Response:
```json
{
  "fraud_probability": 0.0012,
  "is_fraud": false,
  "decision_threshold": 0.90,
  "confidence": "high",
  "timestamp": "2024-04-21T10:30:45Z"
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Info
```bash
curl http://localhost:8000/model-info
```

## Deployment Ready

This system is production-ready because it:
- ✅ Handles real-world imbalanced data
- ✅ Uses business-driven metrics (not accuracy)
- ✅ Tunes threshold for optimal trade-offs
- ✅ Validates all inputs
- ✅ Handles missing values gracefully
- ✅ Has comprehensive test coverage
- ✅ Includes detailed documentation
- ✅ Provides clear error messages
- ✅ Logs all predictions
- ✅ Supports batch processing

## Next Steps

1. Push to GitHub (private repo)
2. Deploy API to production (Docker + Kubernetes)
3. Set up monitoring for model drift
4. Implement feedback loop to retrain model
5. Add authentication/authorization to API
6. Scale to handle millions of transactions
