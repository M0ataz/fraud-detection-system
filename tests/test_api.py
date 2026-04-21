"""
Test suite for fraud detection API.

Tests cover:
- Valid predictions
- Missing values handling
- Invalid inputs
- Edge cases
- Batch predictions
"""

import pytest
import sys
sys.path.insert(0, '/home/ubuntu/fraud-detection-system')

from fastapi.testclient import TestClient
from api.app import app, load_model

# Load model before running tests
load_model()

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True


class TestValidPredictions:
    """Test valid prediction requests."""
    
    def test_legitimate_transaction(self):
        """Test prediction for legitimate transaction."""
        payload = {
            "transaction_amount": 50.0,
            "merchant_risk_score": 0.1,
            "user_age": 45,
            "days_since_account_creation": 1000,
            "transaction_hour": 14,
            "num_transactions_today": 2,
            "device_trust_score": 0.95,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "decision_threshold" in data
        assert "confidence" in data
        assert "timestamp" in data
        
        assert 0 <= data["fraud_probability"] <= 1
        assert isinstance(data["is_fraud"], bool)
        assert data["confidence"] in ["high", "medium", "low"]
    
    def test_suspicious_transaction(self):
        """Test prediction for suspicious transaction."""
        payload = {
            "transaction_amount": 5000.0,
            "merchant_risk_score": 0.9,
            "user_age": 25,
            "days_since_account_creation": 10,
            "transaction_hour": 3,
            "num_transactions_today": 20,
            "device_trust_score": 0.1,
            "failed_login_attempts": 5,
            "international_transaction": 1,
            "card_present": 0,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        # Suspicious transaction should have higher fraud probability
        assert data["fraud_probability"] > 0.3


class TestMissingValues:
    """Test handling of missing values."""
    
    def test_single_missing_value(self):
        """Test prediction with one missing value."""
        payload = {
            "transaction_amount": None,  # Missing
            "merchant_risk_score": 0.2,
            "user_age": 40,
            "days_since_account_creation": 500,
            "transaction_hour": 12,
            "num_transactions_today": 3,
            "device_trust_score": 0.8,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "fraud_probability" in response.json()
    
    def test_multiple_missing_values(self):
        """Test prediction with multiple missing values."""
        payload = {
            "transaction_amount": 100.0,
            "merchant_risk_score": None,  # Missing
            "user_age": None,  # Missing
            "days_since_account_creation": 300,
            "transaction_hour": 10,
            "num_transactions_today": 2,
            "device_trust_score": None,  # Missing
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "fraud_probability" in response.json()
    
    def test_all_missing_values(self):
        """Test prediction with all missing values should fail."""
        payload = {
            "transaction_amount": None,
            "merchant_risk_score": None,
            "user_age": None,
            "days_since_account_creation": None,
            "transaction_hour": None,
            "num_transactions_today": None,
            "device_trust_score": None,
            "failed_login_attempts": None,
            "international_transaction": None,
            "card_present": None,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        assert "missing" in response.json()["detail"].lower()


class TestInvalidInputs:
    """Test invalid input validation."""
    
    def test_invalid_transaction_amount_negative(self):
        """Test rejection of negative transaction amount."""
        payload = {
            "transaction_amount": -50.0,  # Invalid
            "merchant_risk_score": 0.2,
            "user_age": 40,
            "days_since_account_creation": 500,
            "transaction_hour": 12,
            "num_transactions_today": 3,
            "device_trust_score": 0.8,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_transaction_amount_too_high(self):
        """Test rejection of transaction amount > 10000."""
        payload = {
            "transaction_amount": 50000.0,  # Invalid
            "merchant_risk_score": 0.2,
            "user_age": 40,
            "days_since_account_creation": 500,
            "transaction_hour": 12,
            "num_transactions_today": 3,
            "device_trust_score": 0.8,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_invalid_score_out_of_range(self):
        """Test rejection of score outside [0, 1]."""
        payload = {
            "transaction_amount": 100.0,
            "merchant_risk_score": 1.5,  # Invalid
            "user_age": 40,
            "days_since_account_creation": 500,
            "transaction_hour": 12,
            "num_transactions_today": 3,
            "device_trust_score": 0.8,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_invalid_age(self):
        """Test rejection of invalid age."""
        payload = {
            "transaction_amount": 100.0,
            "merchant_risk_score": 0.2,
            "user_age": 150,  # Invalid (> 100)
            "days_since_account_creation": 500,
            "transaction_hour": 12,
            "num_transactions_today": 3,
            "device_trust_score": 0.8,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_invalid_hour(self):
        """Test rejection of invalid hour."""
        payload = {
            "transaction_amount": 100.0,
            "merchant_risk_score": 0.2,
            "user_age": 40,
            "days_since_account_creation": 500,
            "transaction_hour": 25,  # Invalid
            "num_transactions_today": 3,
            "device_trust_score": 0.8,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_invalid_binary_field(self):
        """Test rejection of invalid binary field."""
        payload = {
            "transaction_amount": 100.0,
            "merchant_risk_score": 0.2,
            "user_age": 40,
            "days_since_account_creation": 500,
            "transaction_hour": 12,
            "num_transactions_today": 3,
            "device_trust_score": 0.8,
            "failed_login_attempts": 0,
            "international_transaction": 2,  # Invalid (not 0 or 1)
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestEdgeCases:
    """Test edge cases."""
    
    def test_minimum_valid_values(self):
        """Test with minimum valid values."""
        payload = {
            "transaction_amount": 0.01,
            "merchant_risk_score": 0.0,
            "user_age": 18,
            "days_since_account_creation": 0,
            "transaction_hour": 0,
            "num_transactions_today": 0,
            "device_trust_score": 0.0,
            "failed_login_attempts": 0,
            "international_transaction": 0,
            "card_present": 0,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    def test_maximum_valid_values(self):
        """Test with maximum valid values."""
        payload = {
            "transaction_amount": 10000.0,
            "merchant_risk_score": 1.0,
            "user_age": 100,
            "days_since_account_creation": 10000,
            "transaction_hour": 23,
            "num_transactions_today": 100,
            "device_trust_score": 1.0,
            "failed_login_attempts": 100,
            "international_transaction": 1,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    def test_boundary_values(self):
        """Test boundary values."""
        payload = {
            "transaction_amount": 5000.0,
            "merchant_risk_score": 0.5,
            "user_age": 50,
            "days_since_account_creation": 365,
            "transaction_hour": 12,
            "num_transactions_today": 5,
            "device_trust_score": 0.5,
            "failed_login_attempts": 2,
            "international_transaction": 0,
            "card_present": 1,
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


class TestBatchPredictions:
    """Test batch prediction endpoint."""
    
    def test_batch_prediction_valid(self):
        """Test valid batch prediction."""
        payload = [
            {
                "transaction_amount": 50.0,
                "merchant_risk_score": 0.1,
                "user_age": 45,
                "days_since_account_creation": 1000,
                "transaction_hour": 14,
                "num_transactions_today": 2,
                "device_trust_score": 0.95,
                "failed_login_attempts": 0,
                "international_transaction": 0,
                "card_present": 1,
            },
            {
                "transaction_amount": 5000.0,
                "merchant_risk_score": 0.9,
                "user_age": 25,
                "days_since_account_creation": 10,
                "transaction_hour": 3,
                "num_transactions_today": 20,
                "device_trust_score": 0.1,
                "failed_login_attempts": 5,
                "international_transaction": 1,
                "card_present": 0,
            }
        ]
        
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    
    def test_batch_prediction_empty(self):
        """Test batch prediction with empty list."""
        response = client.post("/predict_batch", json=[])
        assert response.status_code == 400
    
    def test_batch_prediction_too_large(self):
        """Test batch prediction exceeding size limit."""
        payload = [
            {
                "transaction_amount": 100.0,
                "merchant_risk_score": 0.2,
                "user_age": 40,
                "days_since_account_creation": 500,
                "transaction_hour": 12,
                "num_transactions_today": 3,
                "device_trust_score": 0.8,
                "failed_login_attempts": 0,
                "international_transaction": 0,
                "card_present": 1,
            }
        ] * 1001  # 1001 transactions (limit is 1000)
        
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 400


class TestModelInfo:
    """Test model info endpoint."""
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/model-info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "decision_threshold" in data
        assert "features" in data
        assert "num_features" in data
        assert abs(data["decision_threshold"] - 0.9) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
