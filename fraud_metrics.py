#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionMetrics:
    
    def __init__(self, model_path: str = 'best_fraud_model.pkl',
                 scaler_path: str = 'fraud_scaler.pkl',
                 feature_names_path: str = 'feature_names.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feature_names_path)
            print("Model components loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading model components: {e}")
            raise
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      threshold: float = 0.5) -> Dict[str, Any]:
        X_test = X_test[self.feature_names]
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'average_precision': average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'threshold': threshold,
            'n_samples': len(y_test),
            'n_fraud': int(y_test.sum()),
            'n_normal': int((y_test == 0).sum()),
            'fraud_rate': float(y_test.mean()),
            'predicted_fraud': int(y_pred.sum()),
            'predicted_normal': int((y_pred == 0).sum()),
        }
        
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        if len(np.unique(y_test)) > 1:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        if len(np.unique(y_test)) > 1:
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
            metrics['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        
        metrics['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        
        return metrics
    
    def evaluate_on_dataset(self, csv_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if 'Class' not in df.columns:
            raise ValueError("Dataset must have 'Class' column with labels (0=normal, 1=fraud)")
        
        X_test = df.drop('Class', axis=1)
        y_test = df['Class']
        
        return self.evaluate_model(X_test, y_test, threshold)
    
    def find_optimal_threshold(self, X_test: pd.DataFrame, y_test: pd.Series,
                               metric: str = 'f1') -> Tuple[float, float]:
        X_test = X_test[self.feature_names]
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        thresholds = np.arange(0.1, 1.0, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_test, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_test, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_test, y_pred, zero_division=0)
            elif metric == 'fpr':
                cm = confusion_matrix(y_test, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    score = 1 - (fp / (fp + tn) if (fp + tn) > 0 else 1)
                else:
                    continue
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_df.head(top_n)
        else:
            print("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'scores': scores.tolist(),
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def generate_report(self, metrics: Dict[str, Any], 
                       save_path: Optional[str] = None) -> str:
        report = []
        report.append("=" * 80)
        report.append("FRAUD DETECTION MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("CLASSIFICATION METRICS")
        report.append("-" * 80)
        report.append(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        report.append(f"Precision:       {metrics['precision']:.4f}")
        report.append(f"Recall:          {metrics['recall']:.4f}")
        report.append(f"F1 Score:        {metrics['f1_score']:.4f}")
        report.append(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
        report.append(f"Average Precision: {metrics['average_precision']:.4f}")
        report.append("")
        
        report.append("ADDITIONAL METRICS")
        report.append("-" * 80)
        report.append(f"Specificity:     {metrics['specificity']:.4f}")
        report.append(f"Sensitivity:     {metrics['sensitivity']:.4f}")
        report.append(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
        report.append(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
        report.append("")
        
        report.append("DATASET INFORMATION")
        report.append("-" * 80)
        report.append(f"Total Samples:   {metrics['n_samples']:,}")
        report.append(f"Normal:          {metrics['n_normal']:,} ({100*(1-metrics['fraud_rate']):.2f}%)")
        report.append(f"Fraud:           {metrics['n_fraud']:,} ({100*metrics['fraud_rate']:.2f}%)")
        report.append("")
        
        report.append("PREDICTION RESULTS")
        report.append("-" * 80)
        report.append(f"Predicted Normal: {metrics['predicted_normal']:,}")
        report.append(f"Predicted Fraud:  {metrics['predicted_fraud']:,}")
        report.append(f"Threshold:       {metrics['threshold']:.2f}")
        report.append("")
        
        cm = metrics['confusion_matrix']
        report.append("CONFUSION MATRIX")
        report.append("-" * 80)
        report.append(f"                Predicted")
        report.append(f"              Normal  Fraud")
        report.append(f"Actual Normal   {cm['true_negative']:5d}   {cm['false_positive']:5d}")
        report.append(f"Actual Fraud    {cm['false_negative']:5d}   {cm['true_positive']:5d}")
        report.append("")
        
        report.append("DETAILED CLASSIFICATION REPORT")
        report.append("-" * 80)
        class_report = metrics['classification_report']
        for label, scores in class_report.items():
            if isinstance(scores, dict):
                report.append(f"{label}:")
                for metric_name, value in scores.items():
                    if isinstance(value, (int, float)):
                        report.append(f"  {metric_name}: {value:.4f}")
            elif isinstance(scores, (int, float)):
                report.append(f"{label}: {scores:.4f}")
        report.append("")
        
        report.append("=" * 80)
        
        report_str = "\n".join(report)
        
        if save_path:
            metrics_export = {k: v for k, v in metrics.items() 
                            if k not in ['roc_curve', 'pr_curve']}
            with open(save_path.replace('.txt', '.json'), 'w') as f:
                json.dump(metrics_export, f, indent=2)
            
            with open(save_path, 'w') as f:
                f.write(report_str)
            print(f"Report saved to {save_path}")
        
        return report_str
    
    def analyze_predictions(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        analysis = {
            'total_predictions': len(predictions_df),
            'predicted_fraud_count': int(predictions_df['Predicted_Fraud'].sum()),
            'predicted_fraud_rate': float(predictions_df['Predicted_Fraud'].mean()),
        }
        
        if 'Risk_Level' in predictions_df.columns:
            risk_dist = predictions_df['Risk_Level'].value_counts().to_dict()
            analysis['risk_level_distribution'] = {str(k): int(v) for k, v in risk_dist.items()}
        
        if 'Fraud_Probability' in predictions_df.columns:
            probs = predictions_df['Fraud_Probability']
            analysis['probability_stats'] = {
                'mean': float(probs.mean()),
                'median': float(probs.median()),
                'std': float(probs.std()),
                'min': float(probs.min()),
                'max': float(probs.max()),
                'percentile_25': float(probs.quantile(0.25)),
                'percentile_75': float(probs.quantile(0.75)),
                'percentile_95': float(probs.quantile(0.95)),
            }
        
        if 'Actual_Fraud' in predictions_df.columns:
            y_true = predictions_df['Actual_Fraud']
            y_pred = predictions_df['Predicted_Fraud']
            analysis['accuracy'] = float(accuracy_score(y_true, y_pred))
            analysis['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            analysis['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            analysis['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        return analysis


def main():
    import sys
    
    print("=" * 80)
    print("FRAUD DETECTION METRICS CALCULATOR")
    print("=" * 80)
    print()
    
    try:
        metrics_calc = FraudDetectionMetrics()
        
        if len(sys.argv) > 1:
            dataset_path = sys.argv[1]
        else:
            dataset_path = 'creditcard.csv'
        
        print(f"Evaluating model on {dataset_path}...")
        print()
        
        results = metrics_calc.evaluate_on_dataset(dataset_path)
        
        report = metrics_calc.generate_report(results, save_path='fraud_metrics_report.txt')
        print(report)
        
        print("\n" + "=" * 80)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("=" * 80)
        feature_importance = metrics_calc.get_feature_importance(top_n=10)
        if not feature_importance.empty:
            print(feature_importance.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("OPTIMAL THRESHOLD ANALYSIS")
        print("=" * 80)
        df = pd.read_csv(dataset_path)
        X_test = df.drop('Class', axis=1)
        y_test = df['Class']
        
        if len(X_test) > 10000:
            sample_idx = np.random.choice(len(X_test), 10000, replace=False)
            X_sample = X_test.iloc[sample_idx]
            y_sample = y_test.iloc[sample_idx]
        else:
            X_sample = X_test
            y_sample = y_test
        
        opt_threshold_f1, f1_score = metrics_calc.find_optimal_threshold(
            X_sample, y_sample, metric='f1'
        )
        print(f"Optimal threshold (F1): {opt_threshold_f1:.3f} (F1: {f1_score:.4f})")
        
        opt_threshold_recall, recall_score = metrics_calc.find_optimal_threshold(
            X_sample, y_sample, metric='recall'
        )
        print(f"Optimal threshold (Recall): {opt_threshold_recall:.3f} (Recall: {recall_score:.4f})")
        
        print("\nMetrics calculation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
