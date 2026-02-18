#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib

def load_trained_model():
    try:
        best_model = joblib.load('best_fraud_model.pkl')
        scaler = joblib.load('fraud_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        print("Trained model loaded successfully!")
        return best_model, scaler, feature_names
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run fraud_detection.py first to train the model.")
        return None, None, None

def predict_single_transaction(model, scaler, feature_names, transaction_data):
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        df = df[feature_names]
    else:
        df = pd.DataFrame([transaction_data], columns=feature_names)
    
    transaction_scaled = scaler.transform(df)
    fraud_probability = model.predict_proba(transaction_scaled)[:, 1][0]
    fraud_prediction = fraud_probability >= 0.5
    confidence = max(fraud_probability, 1 - fraud_probability)
    
    if fraud_probability < 0.1:
        risk_level = "LOW"
    elif fraud_probability < 0.5:
        risk_level = "MEDIUM"
    elif fraud_probability < 0.8:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    return {
        'fraud_probability': fraud_probability,
        'is_fraud': fraud_prediction,
        'confidence': confidence,
        'risk_level': risk_level,
        'recommendation': get_recommendation(fraud_probability)
    }

def get_recommendation(probability):
    if probability < 0.1:
        return "APPROVE - Low fraud risk"
    elif probability < 0.3:
        return "REVIEW - Monitor transaction"
    elif probability < 0.7:
        return "INVESTIGATE - High fraud risk"
    else:
        return "BLOCK - Very high fraud risk"

def analyze_transaction_features(transaction_data, feature_names):
    print("\n=== Transaction Analysis ===")
    
    try:
        all_models = joblib.load('all_fraud_models.pkl')
        rf_model = all_models['Random Forest']
        
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
            feature = row['feature']
            importance = row['importance']
            value = transaction_data.get(feature, 'N/A')
            print(f"{i+1:2d}. {feature:8s}: {importance:.4f} (Value: {value})")
            
    except Exception as e:
        print(f"Could not analyze features: {e}")

def demo_predictions():
    print("=== Fraud Detection Demo ===\n")
    
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    df = pd.read_csv('creditcard.csv')
    
    normal_transaction = df[df['Class'] == 0].iloc[0].drop('Class').to_dict()
    fraud_transaction = df[df['Class'] == 1].iloc[0].drop('Class').to_dict()
    
    print("Analyzing Sample Transactions:\n")
    
    print("1. NORMAL TRANSACTION:")
    print(f"   Amount: ${normal_transaction['Amount']:.2f}")
    print(f"   Time: {normal_transaction['Time']:.0f} seconds")
    
    result = predict_single_transaction(model, scaler, feature_names, normal_transaction)
    print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Recommendation: {result['recommendation']}")
    
    print("\n2. FRAUD TRANSACTION:")
    print(f"   Amount: ${fraud_transaction['Amount']:.2f}")
    print(f"   Time: {fraud_transaction['Time']:.0f} seconds")
    
    result = predict_single_transaction(model, scaler, feature_names, fraud_transaction)
    print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Recommendation: {result['recommendation']}")
    
    analyze_transaction_features(fraud_transaction, feature_names)

def interactive_prediction():
    print("\n=== Interactive Fraud Prediction ===")
    print("Enter transaction details (press Enter for default values):\n")
    
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    transaction = {}
    
    try:
        amount = input("Transaction Amount: $") or "100.00"
        transaction['Amount'] = float(amount)
        
        time_input = input("Time (seconds since first transaction): ") or "0"
        transaction['Time'] = float(time_input)
        
        print("\nEnter values for V1-V28 features (or press Enter for 0):")
        for i in range(1, 29):
            v_name = f'V{i}'
            value = input(f"{v_name}: ") or "0"
            transaction[v_name] = float(value)
        
        result = predict_single_transaction(model, scaler, feature_names, transaction)
        
        print(f"\nPREDICTION RESULTS:")
        print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Recommendation: {result['recommendation']}")
        
        analyze_transaction_features(transaction, feature_names)
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except KeyboardInterrupt:
        print("\nGoodbye!")

def batch_prediction(file_path):
    print(f"\n=== Batch Prediction from {file_path} ===")
    
    model, scaler, feature_names = load_trained_model()
    if model is None:
        return
    
    try:
        df = pd.read_csv(file_path)
        
        if 'Class' in df.columns:
            df_features = df.drop('Class', axis=1)
            actual_labels = df['Class']
        else:
            df_features = df
            actual_labels = None
        
        for feature in feature_names:
            if feature not in df_features.columns:
                df_features[feature] = 0
        
        df_features = df_features[feature_names]
        
        transactions_scaled = scaler.transform(df_features)
        
        fraud_probabilities = model.predict_proba(transactions_scaled)[:, 1]
        fraud_predictions = (fraud_probabilities >= 0.5).astype(int)
        
        results_df = pd.DataFrame({
            'Transaction_ID': range(len(df_features)),
            'Fraud_Probability': fraud_probabilities,
            'Predicted_Fraud': fraud_predictions,
            'Risk_Level': pd.cut(fraud_probabilities, 
                               bins=[0, 0.1, 0.5, 0.8, 1.0], 
                               labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
        })
        
        if actual_labels is not None:
            results_df['Actual_Fraud'] = actual_labels
            results_df['Correct_Prediction'] = (fraud_predictions == actual_labels)
        
        results_df.to_csv('fraud_predictions.csv', index=False)
        
        print(f"Processed {len(df_features)} transactions")
        print(f"Fraud predictions saved to 'fraud_predictions.csv'")
        
        fraud_count = fraud_predictions.sum()
        print(f"Predicted fraud transactions: {fraud_count} ({fraud_count/len(df_features)*100:.2f}%)")
        
        if actual_labels is not None:
            accuracy = (fraud_predictions == actual_labels).mean()
            print(f"Accuracy: {accuracy:.4f}")
        
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    while True:
        print("\n" + "="*50)
        print("FRAUD DETECTION PREDICTION SYSTEM")
        print("="*50)
        print("1. Demo with sample transactions")
        print("2. Interactive prediction")
        print("3. Batch prediction from CSV")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            demo_predictions()
        elif choice == '2':
            interactive_prediction()
        elif choice == '3':
            file_path = input("Enter CSV file path: ").strip()
            if file_path:
                batch_prediction(file_path)
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()
