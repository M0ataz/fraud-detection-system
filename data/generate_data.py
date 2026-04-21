"""
Generate realistic imbalanced fraud detection dataset.

This script creates a synthetic dataset that mimics real-world fraud patterns:
- Heavy class imbalance (fraud ~1-2% of transactions)
- Missing values in some features
- Realistic transaction patterns
- Temporal patterns (fraud more likely at certain times)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

def generate_fraud_dataset(n_samples=100000, fraud_ratio=0.015):
    """
    Generate imbalanced fraud dataset.
    
    Args:
        n_samples: Total number of transactions
        fraud_ratio: Proportion of fraudulent transactions (realistic: 0.5-2%)
    
    Returns:
        DataFrame with transaction features and fraud labels
    """
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    print(f"Generating dataset: {n_samples} transactions ({fraud_ratio*100:.1f}% fraud)")
    print(f"  - Legitimate: {n_legitimate}")
    print(f"  - Fraudulent: {n_fraud}")
    
    # Normalized probability distributions
    legit_hour_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.08,
                                  0.10, 0.12, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08,
                                  0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01])
    legit_hour_probs = legit_hour_probs / legit_hour_probs.sum()
    
    fraud_hour_probs = np.array([0.08, 0.08, 0.08, 0.08, 0.06, 0.04, 0.03, 0.03,
                                  0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                  0.04, 0.04, 0.04, 0.04, 0.06, 0.08, 0.08, 0.06])
    fraud_hour_probs = fraud_hour_probs / fraud_hour_probs.sum()
    
    # ============ LEGITIMATE TRANSACTIONS ============
    legitimate_data = {
        'transaction_amount': np.random.exponential(scale=150, size=n_legitimate),
        'merchant_risk_score': np.random.beta(2, 5, n_legitimate),  # Most merchants low risk
        'user_age': np.random.normal(loc=45, scale=15, size=n_legitimate),
        'days_since_account_creation': np.random.exponential(scale=500, size=n_legitimate),
        'transaction_hour': np.random.choice(range(24), n_legitimate, p=legit_hour_probs),
        'num_transactions_today': np.random.poisson(lam=3, size=n_legitimate),
        'device_trust_score': np.random.beta(5, 2, n_legitimate),  # Most devices trusted
        'failed_login_attempts': np.random.poisson(lam=0.1, size=n_legitimate),
        'international_transaction': np.random.choice([0, 1], n_legitimate, p=[0.95, 0.05]),
        'card_present': np.random.choice([0, 1], n_legitimate, p=[0.7, 0.3]),
    }
    
    # ============ FRAUDULENT TRANSACTIONS ============
    fraud_data = {
        'transaction_amount': np.random.exponential(scale=300, size=n_fraud),  # Higher amounts
        'merchant_risk_score': np.random.beta(1, 2, n_fraud),  # Higher risk merchants
        'user_age': np.random.normal(loc=35, scale=12, size=n_fraud),  # Younger profile
        'days_since_account_creation': np.random.exponential(scale=100, size=n_fraud),  # Newer accounts
        'transaction_hour': np.random.choice(range(24), n_fraud, p=fraud_hour_probs),
        'num_transactions_today': np.random.poisson(lam=8, size=n_fraud),  # More transactions
        'device_trust_score': np.random.beta(2, 5, n_fraud),  # Lower trust
        'failed_login_attempts': np.random.poisson(lam=1.5, size=n_fraud),  # More failures
        'international_transaction': np.random.choice([0, 1], n_fraud, p=[0.6, 0.4]),  # More international
        'card_present': np.random.choice([0, 1], n_fraud, p=[0.95, 0.05]),  # Mostly online
    }
    
    # Create DataFrames
    df_legitimate = pd.DataFrame(legitimate_data)
    df_fraud = pd.DataFrame(fraud_data)
    
    # Add labels
    df_legitimate['is_fraud'] = 0
    df_fraud['is_fraud'] = 1
    
    # Combine and shuffle
    df = pd.concat([df_legitimate, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # ============ ADD MISSING VALUES (REALISTIC) ============
    # Some features have realistic missing patterns
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'user_age'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'merchant_risk_score'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'device_trust_score'] = np.nan
    
    # ============ CLIP INVALID VALUES ============
    df['transaction_amount'] = df['transaction_amount'].clip(lower=0.01, upper=10000)
    df['user_age'] = df['user_age'].clip(lower=18, upper=100)
    df['days_since_account_creation'] = df['days_since_account_creation'].clip(lower=0)
    df['num_transactions_today'] = df['num_transactions_today'].clip(lower=0, upper=100)
    
    # Ensure scores are in [0, 1]
    for col in ['merchant_risk_score', 'device_trust_score']:
        df[col] = df[col].clip(lower=0, upper=1)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['is_fraud'].value_counts()}")
    print(f"Class imbalance ratio: 1:{(n_legitimate/n_fraud):.1f}")
    
    return df


def split_train_test(df, test_size=0.2, random_state=42):
    """Split data into train/test while preserving class distribution."""
    from sklearn.model_selection import train_test_split
    
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"Test set: {len(X_test)} samples ({y_test.sum()} fraud)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Generate dataset
    df = generate_fraud_dataset(n_samples=100000, fraud_ratio=0.015)
    
    # Save raw data
    df.to_csv('/home/ubuntu/fraud-detection-system/data/transactions.csv', index=False)
    print("\n✓ Saved: data/transactions.csv")
    
    # Split and save
    X_train, X_test, y_train, y_test = split_train_test(df)
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('/home/ubuntu/fraud-detection-system/data/train.csv', index=False)
    test_data.to_csv('/home/ubuntu/fraud-detection-system/data/test.csv', index=False)
    
    print("✓ Saved: data/train.csv")
    print("✓ Saved: data/test.csv")
    
    # Save metadata
    metadata = {
        'total_samples': int(len(df)),
        'fraud_count': int(df['is_fraud'].sum()),
        'fraud_ratio': float(df['is_fraud'].mean()),
        'features': list(X_train.columns),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
    }
    
    import json
    with open('/home/ubuntu/fraud-detection-system/data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Saved: data/metadata.json")
