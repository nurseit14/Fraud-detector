#!/usr/bin/env python3

import os
import sys
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

from langgraph_fraud_workflow import FraudDetectionWorkflow
from conversational_fraud_detector import ConversationalFraudDetector

class LangGraphFraudDetectionSystem:
    
    def __init__(self):
        load_dotenv()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        
        if openai_key:
            print("OpenAI API key found - will use ChatGPT")
        elif dashscope_key:
            print("DashScope API key found - will use Qwen (DashScope)")
        else:
            try:
                import requests
                requests.get("http://localhost:11434/api/tags", timeout=2)
                print("Ollama connection verified - will use Qwen (Ollama)")
            except:
                print("Warning: No LLM service available.")
                print("   Options:")
                print("   1. OpenAI API key in .env file")
                print("   2. DashScope API key in .env file")
                print("   3. Install Ollama and run: ollama pull qwen2:7b")
                print("   You can still use the basic ML model without LangGraph.")
        
        self.workflow = None
        self.chatbot = None
        self.investigator = None
        
        try:
            self.workflow = FraudDetectionWorkflow()
            self.chatbot = ConversationalFraudDetector()
            self.investigator = None
            print("LangGraph components initialized successfully!")
        except Exception as e:
            print(f"LangGraph components not available: {e}")
            print("   Using basic ML model only.")
    
    def analyze_transaction_workflow(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.workflow:
            return self.workflow.analyze_transaction(transaction_data)
        else:
            return {"error": "LangGraph workflow not available"}
    
    def chat_about_fraud(self, user_input: str) -> str:
        if self.chatbot:
            return self.chatbot.chat(user_input)
        else:
            return "Chat interface not available. Please ensure Ollama is running and Qwen model is installed (ollama pull qwen2:7b)."
    
    def investigate_fraud_case(self, case_id: str, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.investigator:
            return self.investigator.investigate_case(case_id, transaction_data)
        else:
            return {"error": "Investigation assistant not available"}
    
    def demo_all_features(self):
        
        print("LangGraph Fraud Detection System Demo")
        print("=" * 60)
        
        df = pd.read_csv('creditcard.csv')
        
        test_transaction = df[df['Class'] == 1].iloc[0].drop('Class').to_dict()
        
        print("\n1. LangGraph Workflow Analysis:")
        print("-" * 40)
        if self.workflow:
            result = self.analyze_transaction_workflow(test_transaction)
            print(f"Fraud Probability: {result['fraud_prediction']['fraud_probability']:.4f}")
            print(f"Risk Assessment: {result['risk_assessment'][:100]}...")
            print(f"Explanation: {result['explanation'][:100]}...")
        else:
            print("Workflow not available")
        
        print("\n2. Conversational Interface:")
        print("-" * 40)
        if self.chatbot:
            response = self.chat_about_fraud("Analyze this transaction: amount $100, time 1000 seconds")
            print(f"Response: {response[:200]}...")
        else:
            print("Chat interface not available")
        
        print("\n3. Investigation Assistant:")
        print("-" * 40)
        print("Investigation assistant not available (file removed)")
        
        print("\nDemo completed!")

def main():
    system = LangGraphFraudDetectionSystem()
    
    while True:
        print("\n" + "="*60)
        print("LANGGRAPH FRAUD DETECTION SYSTEM")
        print("="*60)
        print("1. Run LangGraph Workflow Analysis")
        print("2. Start Conversational Interface")
        print("3. Launch Investigation Assistant")
        print("4. Demo All Features")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            print("\nLangGraph Workflow Analysis")
            print("-" * 40)
            
            try:
                amount = float(input("Transaction Amount ($): "))
                time = float(input("Time (seconds): "))
                
                transaction_data = {
                    "Amount": amount,
                    "Time": time
                }
                
                for i in range(1, 29):
                    transaction_data[f"V{i}"] = 0
                
                result = system.analyze_transaction_workflow(transaction_data)
                
                if "error" not in result:
                    print(f"\nAnalysis Results:")
                    print(f"Fraud Probability: {result['fraud_prediction']['fraud_probability']:.4f}")
                    print(f"Risk Assessment: {result['risk_assessment']}")
                    print(f"Explanation: {result['explanation']}")
                    print(f"Recommendations: {result['recommendations'][:3]}")
                else:
                    print(f"{result['error']}")
                    
            except ValueError:
                print("Please enter valid numeric values.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            print("\nStarting Conversational Interface...")
            print("Type 'quit' to exit the chat.")
            
            if system.chatbot:
                while True:
                    try:
                        user_input = input("\nYou: ").strip()
                        if user_input.lower() in ['quit', 'exit']:
                            break
                        response = system.chat_about_fraud(user_input)
                        print(f"Assistant: {response}")
                    except KeyboardInterrupt:
                        break
            else:
                print("Chat interface not available. Please ensure Ollama is running and Qwen model is installed (ollama pull qwen2:7b).")
        
        elif choice == '3':
            print("\nStarting Investigation Assistant...")
            
            if system.investigator:
                try:
                    amount = float(input("Transaction Amount ($): "))
                    time = float(input("Time (seconds): "))
                    
                    transaction_data = {
                        "Amount": amount,
                        "Time": time
                    }
                    
                    for i in range(1, 29):
                        transaction_data[f"V{i}"] = 0
                    
                    case_id = f"CASE-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                    result = system.investigate_fraud_case(case_id, transaction_data)
                    
                    print(f"\nInvestigation Results:")
                    print(f"Case ID: {result['case_id']}")
                    print(f"Risk Level: {result['fraud_analysis']['risk_level']}")
                    print(f"Case Status: {result['case_status']}")
                    print(f"Investigation Steps: {len(result['investigation_steps'])}")
                    print(f"Evidence Collected: {len(result['evidence_collected'])}")
                    
                except ValueError:
                    print("Please enter valid numeric values.")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Investigation assistant not available. Please ensure Ollama is running and Qwen model is installed (ollama pull qwen2:7b).")
        
        elif choice == '4':
            system.demo_all_features()
        
        elif choice == '5':
            print("Goodbye! Stay safe from fraud!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()
