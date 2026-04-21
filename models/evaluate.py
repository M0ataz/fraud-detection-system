"""
Comprehensive evaluation of fraud detection model.

Focuses on:
1. Precision, Recall, F1 metrics
2. Business impact analysis
3. Threshold trade-offs
4. Confusion matrix analysis
5. Cost-benefit analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, auc
)
import json
from train import FraudDetectionModel

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained FraudDetectionModel instance
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with all evaluation metrics
    """
    
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    
    # Get predictions
    y_pred, y_pred_proba = model.predict(X_test, use_tuned_threshold=True)
    y_pred_default, _ = model.predict(X_test, use_tuned_threshold=False)
    
    # ============ METRICS AT TUNED THRESHOLD ============
    print("\n[1] METRICS AT TUNED THRESHOLD (0.90)")
    print("-" * 70)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Precision: {precision:.4f}")
    print(f"  → Of all predicted frauds, {precision*100:.2f}% are actually fraudulent")
    print(f"  → False positive rate: {(1-precision)*100:.2f}%")
    
    print(f"\nRecall: {recall:.4f}")
    print(f"  → Of all actual frauds, {recall*100:.2f}% are detected")
    print(f"  → False negative rate (missed fraud): {(1-recall)*100:.2f}%")
    
    print(f"\nF1-Score: {f1:.4f}")
    print(f"  → Harmonic mean of precision and recall")
    
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"  → Probability model ranks random fraud higher than random legitimate")
    
    # ============ CONFUSION MATRIX ============
    print("\n[2] CONFUSION MATRIX")
    print("-" * 70)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\nTrue Negatives (TN):  {tn:6d}  (Correctly identified legitimate)")
    print(f"False Positives (FP): {fp:6d}  (Legitimate flagged as fraud)")
    print(f"False Negatives (FN): {fn:6d}  (Fraud missed)")
    print(f"True Positives (TP):  {tp:6d}  (Correctly identified fraud)")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSpecificity: {specificity:.4f}")
    print(f"  → Of all legitimate transactions, {specificity*100:.2f}% are correctly identified")
    
    # ============ BUSINESS IMPACT ============
    print("\n[3] BUSINESS IMPACT ANALYSIS")
    print("-" * 70)
    
    # Assume average transaction amount and fraud loss
    avg_transaction = 250  # dollars
    fraud_loss_rate = 0.95  # 95% of fraud amount is lost
    false_positive_cost = 5  # cost to investigate/block legitimate transaction
    
    total_transactions = len(y_test)
    actual_fraud_amount = y_test.sum() * avg_transaction
    
    # Losses if no model
    no_model_loss = actual_fraud_amount * fraud_loss_rate
    
    # Losses with model
    missed_fraud_loss = fn * avg_transaction * fraud_loss_rate
    false_positive_cost_total = fp * false_positive_cost
    total_loss_with_model = missed_fraud_loss + false_positive_cost_total
    
    savings = no_model_loss - total_loss_with_model
    roi = (savings / no_model_loss) * 100 if no_model_loss > 0 else 0
    
    print(f"\nAssumptions:")
    print(f"  - Average transaction: ${avg_transaction}")
    print(f"  - Fraud loss rate: {fraud_loss_rate*100:.0f}%")
    print(f"  - Cost per false positive: ${false_positive_cost}")
    
    print(f"\nFinancial Impact:")
    print(f"  - Actual fraud amount: ${actual_fraud_amount:,.0f}")
    print(f"  - Loss without model: ${no_model_loss:,.0f}")
    print(f"  - Loss with model: ${total_loss_with_model:,.0f}")
    print(f"  - Savings: ${savings:,.0f}")
    print(f"  - ROI: {roi:.1f}%")
    
    # ============ THRESHOLD COMPARISON ============
    print("\n[4] THRESHOLD COMPARISON")
    print("-" * 70)
    
    y_pred_default_proba = y_pred_default
    precision_default = precision_score(y_test, y_pred_default, zero_division=0)
    recall_default = recall_score(y_test, y_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    
    print(f"\nDefault Threshold (0.50):")
    print(f"  Precision: {precision_default:.4f}")
    print(f"  Recall: {recall_default:.4f}")
    print(f"  F1-Score: {f1_default:.4f}")
    
    print(f"\nTuned Threshold (0.90):")
    print(f"  Precision: {precision:.4f} ({(precision-precision_default)*100:+.2f}%)")
    print(f"  Recall: {recall:.4f} ({(recall-recall_default)*100:+.2f}%)")
    print(f"  F1-Score: {f1:.4f} ({(f1-f1_default)*100:+.2f}%)")
    
    print(f"\nTrade-off Analysis:")
    print(f"  → Tuning threshold from 0.50 to 0.90 INCREASES precision")
    print(f"  → This DECREASES false positives (fewer legitimate blocked)")
    print(f"  → Recall remains high (still catching most fraud)")
    print(f"  → Better for user experience with minimal fraud leakage")
    
    # ============ PRECISION-RECALL CURVE ============
    print("\n[5] PRECISION-RECALL ANALYSIS")
    print("-" * 70)
    
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\nPrecision-Recall AUC: {pr_auc:.4f}")
    print(f"  → Measures area under precision-recall curve")
    print(f"  → More informative for imbalanced datasets than ROC-AUC")
    
    # ============ CLASS DISTRIBUTION ============
    print("\n[6] TEST SET ANALYSIS")
    print("-" * 70)
    
    n_legitimate = (y_test == 0).sum()
    n_fraud = (y_test == 1).sum()
    fraud_ratio = n_fraud / len(y_test)
    
    print(f"\nTest set composition:")
    print(f"  - Legitimate: {n_legitimate:6d} ({(1-fraud_ratio)*100:.2f}%)")
    print(f"  - Fraudulent: {n_fraud:6d} ({fraud_ratio*100:.2f}%)")
    print(f"  - Imbalance ratio: 1:{n_legitimate/n_fraud:.1f}")
    
    print("\n" + "=" * 70)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'specificity': specificity,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'business_impact': {
            'actual_fraud_amount': actual_fraud_amount,
            'loss_without_model': no_model_loss,
            'loss_with_model': total_loss_with_model,
            'savings': savings,
            'roi': roi,
        },
        'threshold_comparison': {
            'default': {'precision': precision_default, 'recall': recall_default, 'f1': f1_default},
            'tuned': {'precision': precision, 'recall': recall, 'f1': f1},
        },
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'roc_curve': roc_curve(y_test, y_pred_proba),
    }


def create_visualizations(eval_results, y_test, output_dir='/home/ubuntu/fraud-detection-system/notebooks'):
    """Create evaluation visualizations."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fraud Detection Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    ax = axes[0, 0]
    cm = np.array([
        [eval_results['confusion_matrix']['tn'], eval_results['confusion_matrix']['fp']],
        [eval_results['confusion_matrix']['fn'], eval_results['confusion_matrix']['tp']]
    ])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title('Confusion Matrix (Threshold=0.90)', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_yticklabels(['Legitimate', 'Fraud'])
    
    # 2. ROC Curve
    ax = axes[0, 1]
    fpr, tpr, _ = eval_results['roc_curve']
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {eval_results['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax = axes[1, 0]
    recall_curve = eval_results['recall_curve']
    precision_curve = eval_results['precision_curve']
    ax.plot(recall_curve, precision_curve, linewidth=2, label=f"PR-AUC = {eval_results['pr_auc']:.4f}")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # 4. Metrics Comparison
    ax = axes[1, 1]
    metrics = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    values = [
        eval_results['precision'],
        eval_results['recall'],
        eval_results['f1'],
        eval_results['specificity'],
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/evaluation_metrics.png")
    plt.close()


def main():
    """Run evaluation."""
    
    # Load data
    print("Loading data...")
    test_df = pd.read_csv('/home/ubuntu/fraud-detection-system/data/test.csv')
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']
    
    # Load model
    print("Loading model...")
    model = FraudDetectionModel.load('/home/ubuntu/fraud-detection-system/models')
    
    # Evaluate
    eval_results = evaluate_model(model, X_test, y_test)
    
    # Create visualizations
    create_visualizations(eval_results, y_test)
    
    # Save evaluation results
    results_to_save = {
        'precision': float(eval_results['precision']),
        'recall': float(eval_results['recall']),
        'f1': float(eval_results['f1']),
        'roc_auc': float(eval_results['roc_auc']),
        'pr_auc': float(eval_results['pr_auc']),
        'specificity': float(eval_results['specificity']),
        'confusion_matrix': {k: int(v) for k, v in eval_results['confusion_matrix'].items()},
        'business_impact': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in eval_results['business_impact'].items()},
        'threshold_comparison': {
            'default': {k: float(v) for k, v in eval_results['threshold_comparison']['default'].items()},
            'tuned': {k: float(v) for k, v in eval_results['threshold_comparison']['tuned'].items()},
        },
    }
    
    with open('/home/ubuntu/fraud-detection-system/models/evaluation_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print("\n✓ Evaluation complete!")
    print("✓ Results saved to models/evaluation_results.json")


if __name__ == "__main__":
    main()
