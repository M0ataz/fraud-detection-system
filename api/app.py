"""
Production FastAPI application for fraud detection.

Features:
- POST /predict endpoint for fraud detection
- Comprehensive input validation and error handling
- Missing value handling
- Invalid input rejection
- Detailed error messages
- Request/response logging
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import sys
import os

# Add models directory to path
sys.path.insert(0, '/home/ubuntu/fraud-detection-system')

from models.train import FraudDetectionModel

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ FASTAPI APP ============
app = FastAPI(
    title="Fraud Detection API",
    description="Production-grade fraud detection system with threshold tuning",
    version="1.0.0"
)

# ============ GLOBAL MODEL ============
model = None

def load_model():
    """Load model on startup."""
    global model
    try:
        model = FraudDetectionModel.load('/home/ubuntu/fraud-detection-system/models')
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    load_model()


# ============ REQUEST/RESPONSE MODELS ============
class TransactionRequest(BaseModel):
    """
    Transaction data for fraud prediction.
    
    All fields are required. Missing values should be represented as null
    and will be imputed using median strategy.
    """
    
    transaction_amount: Optional[float] = Field(
        None, 
        description="Transaction amount in dollars (0.01 - 10000)",
        example=150.50
    )
    merchant_risk_score: Optional[float] = Field(
        None,
        description="Merchant risk score (0.0 - 1.0)",
        example=0.15
    )
    user_age: Optional[float] = Field(
        None,
        description="User age in years (18 - 100)",
        example=35
    )
    days_since_account_creation: Optional[float] = Field(
        None,
        description="Days since account creation (0+)",
        example=500
    )
    transaction_hour: Optional[int] = Field(
        None,
        description="Hour of transaction (0-23)",
        example=14
    )
    num_transactions_today: Optional[int] = Field(
        None,
        description="Number of transactions today (0+)",
        example=3
    )
    device_trust_score: Optional[float] = Field(
        None,
        description="Device trust score (0.0 - 1.0)",
        example=0.85
    )
    failed_login_attempts: Optional[int] = Field(
        None,
        description="Failed login attempts (0+)",
        example=0
    )
    international_transaction: Optional[int] = Field(
        None,
        description="Is international transaction (0 or 1)",
        example=0
    )
    card_present: Optional[int] = Field(
        None,
        description="Was card physically present (0 or 1)",
        example=1
    )
    
    @validator('transaction_amount')
    def validate_amount(cls, v):
        if v is not None and (v < 0.01 or v > 10000):
            raise ValueError('transaction_amount must be between 0.01 and 10000')
        return v
    
    @validator('merchant_risk_score', 'device_trust_score')
    def validate_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Score must be between 0 and 1')
        return v
    
    @validator('user_age')
    def validate_age(cls, v):
        if v is not None and (v < 18 or v > 100):
            raise ValueError('user_age must be between 18 and 100')
        return v
    
    @validator('days_since_account_creation')
    def validate_days(cls, v):
        if v is not None and v < 0:
            raise ValueError('days_since_account_creation must be >= 0')
        return v
    
    @validator('transaction_hour')
    def validate_hour(cls, v):
        if v is not None and (v < 0 or v > 23):
            raise ValueError('transaction_hour must be between 0 and 23')
        return v
    
    @validator('num_transactions_today', 'failed_login_attempts')
    def validate_count(cls, v):
        if v is not None and v < 0:
            raise ValueError('Count must be >= 0')
        return v
    
    @validator('international_transaction', 'card_present')
    def validate_binary(cls, v):
        if v is not None and v not in [0, 1]:
            raise ValueError('Binary field must be 0 or 1')
        return v


class PredictionResponse(BaseModel):
    """Fraud prediction response."""
    
    fraud_probability: float = Field(
        ...,
        description="Probability of fraud (0.0 - 1.0)",
        example=0.15
    )
    is_fraud: bool = Field(
        ...,
        description="Binary fraud prediction using tuned threshold",
        example=False
    )
    decision_threshold: float = Field(
        ...,
        description="Decision threshold used for binary prediction",
        example=0.90
    )
    confidence: str = Field(
        ...,
        description="Confidence level of prediction",
        example="high"
    )
    timestamp: str = Field(
        ...,
        description="Prediction timestamp",
        example="2024-04-21T10:30:45Z"
    )


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


# ============ ENDPOINTS ============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest):
    """
    Predict fraud probability for a transaction.
    
    This endpoint:
    1. Validates all inputs
    2. Handles missing values via imputation
    3. Preprocesses features
    4. Generates fraud probability
    5. Applies tuned threshold for binary decision
    
    Args:
        request: Transaction data
    
    Returns:
        Fraud prediction with probability and binary decision
    
    Raises:
        HTTPException: If input validation fails or model unavailable
    """
    
    try:
        # Check model is loaded
        if model is None:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please try again later."
            )
        
        # Convert request to DataFrame
        request_dict = request.model_dump()
        X = pd.DataFrame([request_dict])
        
        logger.info(f"Processing prediction request: {request_dict}")
        
        # ============ INPUT VALIDATION ============
        # Check for all null values
        if X.isnull().all().all():
            raise HTTPException(
                status_code=400,
                detail="All input features are missing. Provide at least some feature values."
            )
        
        # Get fraud probability
        fraud_probability = float(model.predict_proba(X)[0])
        
        # Get binary prediction using tuned threshold
        is_fraud = fraud_probability >= model.threshold
        
        # Determine confidence level
        prob_distance = abs(fraud_probability - model.threshold)
        if prob_distance > 0.3:
            confidence = "high"
        elif prob_distance > 0.15:
            confidence = "medium"
        else:
            confidence = "low"
        
        response = PredictionResponse(
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            decision_threshold=model.threshold,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"Prediction: fraud={is_fraud}, prob={fraud_probability:.4f}")
        
        return response
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@app.post("/predict_batch")
async def predict_batch(requests: list[TransactionRequest]):
    """
    Batch prediction endpoint for multiple transactions.
    
    Args:
        requests: List of transaction requests
    
    Returns:
        List of predictions
    """
    
    if not requests:
        raise HTTPException(
            status_code=400,
            detail="Empty request list"
        )
    
    if len(requests) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 1000 transactions"
        )
    
    try:
        predictions = []
        for req in requests:
            pred = await predict(req)
            predictions.append(pred)
        
        logger.info(f"Batch prediction: {len(predictions)} transactions processed")
        
        return {
            "count": len(predictions),
            "predictions": predictions,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing batch predictions"
        )


@app.get("/model-info")
async def model_info():
    """Get model information and configuration."""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    return {
        "model_type": "XGBoost with SMOTE",
        "decision_threshold": model.threshold,
        "features": model.feature_names,
        "num_features": len(model.feature_names),
        "preprocessing": {
            "missing_value_strategy": "median imputation",
            "scaling": "StandardScaler (mean=0, std=1)"
        },
        "imbalance_handling": {
            "method": "SMOTE + scale_pos_weight",
            "description": "Synthetic oversampling + native XGBoost class weighting"
        }
    }


# ============ EXCEPTION HANDLERS ============

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return {
        "error": "Validation error",
        "details": str(exc),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
