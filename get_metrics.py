#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fraud_metrics import FraudDetectionMetrics
import sys
import os

def create_visualizations(metrics, metrics_calc, dataset_path='creditcard.csv'):
    print("\nGenerating visualizations...")
    
    output_dir = 'metrics_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
    
    fig_count = 0
    
    if 'roc_curve' in metrics and metrics['roc_curve']:
        fig, ax = plt.subplots(figsize=(10, 8))
        roc_data = metrics['roc_curve']
        ax.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2, 
                label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Fraud Detection Model', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        print(f"  Saved: {output_dir}/roc_curve.png")
    
    if 'pr_curve' in metrics and metrics['pr_curve']:
        fig, ax = plt.subplots(figsize=(10, 8))
        pr_data = metrics['pr_curve']
        ax.plot(pr_data['recall'], pr_data['precision'], linewidth=2,
                label=f"PR Curve (AP = {metrics['average_precision']:.4f})")
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Fraud Detection Model', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        print(f"  Saved: {output_dir}/precision_recall_curve.png")
    
    cm = metrics['confusion_matrix']
    cm_matrix = np.array([
        [cm['true_negative'], cm['false_positive']],
        [cm['false_negative'], cm['true_positive']]
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'],
                cbar_kws={'label': 'Count'},
                ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Fraud Detection Model', fontsize=14, fontweight='bold')
    
    total = cm_matrix.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm_matrix[i, j] / total) * 100
            ax.text(j + 0.5, i + 0.7, f'\n({percentage:.1f}%)', 
                   ha='center', va='top', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    print(f"  Saved: {output_dir}/confusion_matrix.png")
    
    feature_importance = metrics_calc.get_feature_importance(top_n=15)
    if not feature_importance.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
        bars = ax.barh(range(len(feature_importance)), 
                      feature_importance['importance'].values,
                      color=colors)
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'].values, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (idx, row) in enumerate(feature_importance.iterrows()):
            ax.text(row['importance'] + 0.001, i, f"{row['importance']:.4f}",
                   va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        fig_count += 1
        print(f"  Saved: {output_dir}/feature_importance.png")
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    metrics_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_names)))
    bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (name, value) in enumerate(zip(metrics_names, metrics_values)):
        ax.text(i, value + 0.02, f'{value:.4f}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i, value / 2, f'{value*100:.1f}%', 
               ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    print(f"  Saved: {output_dir}/metrics_comparison.png")
    
    cm_dict = metrics['confusion_matrix']
    cm_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    cm_values = [cm_dict['true_negative'], cm_dict['false_positive'], 
                cm_dict['false_negative'], cm_dict['true_positive']]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_cm = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    wedges, texts, autotexts = ax.pie(cm_values, labels=cm_labels, autopct='%1.1f%%',
                                     colors=colors_cm, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
    
    ax.set_title('Confusion Matrix Distribution', fontsize=14, fontweight='bold', pad=20)
    
    total_samples = sum(cm_values)
    legend_text = [f'{label}: {value:,} ({value/total_samples*100:.1f}%)' 
                   for label, value in zip(cm_labels, cm_values)]
    ax.legend(wedges, legend_text, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig_count += 1
    print(f"  Saved: {output_dir}/confusion_matrix_pie.png")
    
    print(f"\nTotal visualizations saved: {fig_count}")
    print(f"All plots saved in: {output_dir}/")
    
    return output_dir

def main():
    print("=" * 80)
    print("FRAUD DETECTION METRICS")
    print("=" * 80)
    print()
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = 'creditcard.csv'
    
    try:
        print("Loading model components...")
        metrics_calc = FraudDetectionMetrics()
        print()
        
        print(f"Evaluating on {dataset_path}...")
        print()
        metrics = metrics_calc.evaluate_on_dataset(dataset_path)
        
        report = metrics_calc.generate_report(
            metrics, 
            save_path='fraud_metrics_report.txt'
        )
        print(report)
        
        print("\n" + "=" * 80)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("=" * 80)
        feature_importance = metrics_calc.get_feature_importance(top_n=10)
        if not feature_importance.empty:
            print(feature_importance.to_string(index=False))
        
        try:
            pred_df = pd.read_csv('fraud_predictions.csv')
            print("\n" + "=" * 80)
            print("PREDICTION ANALYSIS")
            print("=" * 80)
            analysis = metrics_calc.analyze_predictions(pred_df)
            
            print(f"Total Predictions: {analysis['total_predictions']:,}")
            print(f"Predicted Fraud: {analysis['predicted_fraud_count']:,} ({analysis['predicted_fraud_rate']*100:.2f}%)")
            
            if 'risk_level_distribution' in analysis:
                print("\nRisk Level Distribution:")
                for level, count in analysis['risk_level_distribution'].items():
                    print(f"  {level}: {count:,}")
            
            if 'probability_stats' in analysis:
                stats = analysis['probability_stats']
                print("\nProbability Statistics:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Median: {stats['median']:.4f}")
                print(f"  Std: {stats['std']:.4f}")
                print(f"  Min: {stats['min']:.4f}")
                print(f"  Max: {stats['max']:.4f}")
            
            if 'accuracy' in analysis:
                print("\nPrediction Accuracy:")
                print(f"  Accuracy: {analysis['accuracy']:.4f}")
                print(f"  Precision: {analysis['precision']:.4f}")
                print(f"  Recall: {analysis['recall']:.4f}")
                print(f"  F1 Score: {analysis['f1_score']:.4f}")
        except FileNotFoundError:
            print("\nTip: Run batch predictions to analyze prediction results")
        
        output_dir = create_visualizations(metrics, metrics_calc, dataset_path)
        
        print("\n" + "=" * 80)
        print("All metrics calculated successfully!")
        print("=" * 80)
        print(f"Full report saved to: fraud_metrics_report.txt")
        print(f"Metrics JSON saved to: fraud_metrics_report.json")
        print(f"Visualizations saved to: {output_dir}/")
        print(f"  - roc_curve.png")
        print(f"  - precision_recall_curve.png")
        print(f"  - confusion_matrix.png")
        print(f"  - confusion_matrix_pie.png")
        print(f"  - feature_importance.png")
        print(f"  - metrics_comparison.png")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure:")
        print("  - Model files exist (best_fraud_model.pkl, fraud_scaler.pkl, feature_names.pkl)")
        print(f"  - Dataset file exists ({dataset_path})")
    except ImportError as e:
        print(f"Error: Missing required library - {e}")
        print("Please install matplotlib and seaborn:")
        print("  pip install matplotlib seaborn")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
