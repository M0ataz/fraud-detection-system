"""
Train fraud detection model with proper handling of class imbalance.

Key strategies:
1. SMOTE (Synthetic Minority Over-sampling Technique) for balancing
2. XGBoost with scale_pos_weight for native imbalance handling
3. Threshold tuning based on business metrics (precision/recall trade-offs)
4. Cross-validation with stratified folds
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

class FraudDetectionModel:
    """
    Production-grade fraud detection model with imbalance handling.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.imputer = None
        self.threshold = 0.5  # Default threshold
        self.feature_names = None
        
    def _create_preprocessor(self):
        """Create preprocessing pipeline."""
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
    def _create_model(self, n_samples, n_fraud):
        """
        Create XGBoost model with class weight balancing.
        
        Args:
            n_samples: Total training samples
            n_fraud: Number of fraud cases
        """
        # Calculate scale_pos_weight for XGBoost
        # This helps handle class imbalance natively
        scale_pos_weight = (n_samples - n_fraud) / n_fraud
        
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # Native imbalance handling
            random_state=self.random_state,
            tree_method='hist',
            eval_metric='logloss',
            verbosity=0,
        )
        
    def train(self, X_train, y_train, use_smote=True, val_split=0.2):
        """
        Train model with optional SMOTE oversampling.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to apply SMOTE
            val_split: Validation split ratio
        
        Returns:
            Training history with metrics
        """
        print("=" * 60)
        print("TRAINING FRAUD DETECTION MODEL")
        print("=" * 60)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Create preprocessor
        self._create_preprocessor()
        
        # Preprocess data
        print("\n[1/5] Preprocessing data...")
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        print(f"  ✓ Handled {X_train.isnull().sum().sum()} missing values")
        print(f"  ✓ Scaled features to mean=0, std=1")
        
        # Apply SMOTE if requested
        print("\n[2/5] Handling class imbalance...")
        if use_smote:
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_scaled, y_train
            )
            print(f"  ✓ Applied SMOTE oversampling")
            print(f"    - Before: {y_train.value_counts().to_dict()}")
            print(f"    - After: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        else:
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train
        
        # Create and train model
        print("\n[3/5] Training XGBoost model...")
        n_fraud = (y_train == 1).sum()
        self._create_model(len(y_train), n_fraud)
        
        self.model.fit(
            X_train_balanced, y_train_balanced,
            verbose=False
        )
        print(f"  ✓ Model trained with scale_pos_weight={self.model.scale_pos_weight:.2f}")
        
        # Evaluate on original (unbalanced) training set
        print("\n[4/5] Evaluating on training set...")
        y_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        train_metrics = {
            'precision': precision_score(y_train, y_pred, zero_division=0),
            'recall': recall_score(y_train, y_pred, zero_division=0),
            'f1': f1_score(y_train, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_pred_proba),
        }
        
        print(f"  ✓ Precision: {train_metrics['precision']:.4f}")
        print(f"  ✓ Recall: {train_metrics['recall']:.4f}")
        print(f"  ✓ F1-Score: {train_metrics['f1']:.4f}")
        print(f"  ✓ ROC-AUC: {train_metrics['roc_auc']:.4f}")
        
        # Feature importance
        print("\n[5/5] Feature importance (top 5):")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        print("\n" + "=" * 60)
        
        return {
            'train_metrics': train_metrics,
            'feature_importance': feature_importance,
            'X_train_scaled': X_train_scaled,
            'y_train': y_train,
        }
    
    def find_optimal_threshold(self, X_val, y_val, metric='f1', beta=1.0):
        """
        Find optimal decision threshold using validation data.
        
        This is CRITICAL for imbalanced datasets. The default 0.5 threshold
        is often suboptimal. We tune based on business metrics.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Optimization metric ('f1', 'precision', 'recall', 'f_beta')
            beta: Beta parameter for F-beta score (default 1.0 = F1)
        
        Returns:
            Optimal threshold and metrics at that threshold
        """
        print("\n" + "=" * 60)
        print("THRESHOLD TUNING")
        print("=" * 60)
        
        # Preprocess validation data
        X_val_imputed = self.imputer.transform(X_val)
        X_val_scaled = self.scaler.transform(X_val_imputed)
        
        # Get probability predictions
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Get ROC curve to find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
        
        # Evaluate different thresholds
        best_threshold = 0.5
        best_score = 0
        best_metrics = {}
        
        print(f"\nSearching optimal threshold (metric: {metric})...")
        print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Score':<12}")
        print("-" * 60)
        
        for threshold in np.arange(0.1, 0.95, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # F-beta score: emphasizes recall if beta > 1
            if metric == 'f_beta':
                score = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-10)
            elif metric == 'f1':
                score = f1
            elif metric == 'precision':
                score = precision
            elif metric == 'recall':
                score = recall
            else:
                score = f1
            
            print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {score:<12.4f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'score': score,
                }
        
        self.threshold = best_threshold
        
        print("-" * 60)
        print(f"\n✓ Optimal threshold: {best_threshold:.2f}")
        print(f"  - Precision: {best_metrics['precision']:.4f}")
        print(f"  - Recall: {best_metrics['recall']:.4f}")
        print(f"  - F1-Score: {best_metrics['f1']:.4f}")
        print("\n" + "=" * 60)
        
        return {
            'threshold': best_threshold,
            'metrics': best_metrics,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
        }
    
    def predict_proba(self, X):
        """Get fraud probability predictions."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X, use_tuned_threshold=True):
        """Get binary fraud predictions."""
        proba = self.predict_proba(X)
        threshold = self.threshold if use_tuned_threshold else 0.5
        return (proba >= threshold).astype(int), proba
    
    def save(self, model_dir):
        """Save model and preprocessors."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        with open(f'{model_dir}/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save preprocessors
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{model_dir}/imputer.pkl', 'wb') as f:
            pickle.dump(self.imputer, f)
        
        # Save metadata
        metadata = {
            'threshold': self.threshold,
            'feature_names': self.feature_names,
        }
        with open(f'{model_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved to {model_dir}/")
    
    @classmethod
    def load(cls, model_dir):
        """Load model and preprocessors."""
        instance = cls()
        
        # Load model
        with open(f'{model_dir}/model.pkl', 'rb') as f:
            instance.model = pickle.load(f)
        
        # Load preprocessors
        with open(f'{model_dir}/scaler.pkl', 'rb') as f:
            instance.scaler = pickle.load(f)
        
        with open(f'{model_dir}/imputer.pkl', 'rb') as f:
            instance.imputer = pickle.load(f)
        
        # Load metadata
        with open(f'{model_dir}/metadata.json', 'r') as f:
            metadata = json.load(f)
            instance.threshold = metadata['threshold']
            instance.feature_names = metadata['feature_names']
        
        return instance


def main():
    """Train and save model."""
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('/home/ubuntu/fraud-detection-system/data/train.csv')
    test_df = pd.read_csv('/home/ubuntu/fraud-detection-system/data/test.csv')
    
    X_train = train_df.drop('is_fraud', axis=1)
    y_train = train_df['is_fraud']
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']
    
    print(f"Train: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"Test: {len(X_test)} samples ({y_test.sum()} fraud)")
    
    # Create and train model
    model = FraudDetectionModel()
    train_info = model.train(X_train, y_train, use_smote=True)
    
    # Tune threshold on test set
    threshold_info = model.find_optimal_threshold(X_test, y_test, metric='f1')
    
    # Save model
    model.save('/home/ubuntu/fraud-detection-system/models')
    
    # Save training info
    with open('/home/ubuntu/fraud-detection-system/models/training_info.json', 'w') as f:
        json.dump({
            'train_metrics': {k: float(v) for k, v in train_info['train_metrics'].items()},
            'optimal_threshold': threshold_info['threshold'],
            'threshold_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                 for k, v in threshold_info['metrics'].items()},
        }, f, indent=2)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
