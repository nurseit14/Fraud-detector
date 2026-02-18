#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

class FraudAnalysisState(TypedDict):
    transaction_data: Dict[str, Any]
    fraud_prediction: Dict[str, Any]
    risk_assessment: str
    explanation: str
    recommendations: List[str]
    investigation_steps: List[str]
    messages: List[Any]

class FraudDetectionWorkflow:
    
    def __init__(self):
        self.model = joblib.load('best_fraud_model.pkl')
        self.scaler = joblib.load('fraud_scaler.pkl')
        self.feature_names = joblib.load('feature_names.pkl')
        
        openai_key = os.getenv("OPENAI_API_KEY")
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        
        if openai_key:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=openai_key
            )
            print("OpenAI (ChatGPT) initialized successfully!")
        
        elif dashscope_key:
            from langchain_community.chat_models import ChatTongyi
            self.llm = ChatTongyi(
                model="qwen-turbo",
                temperature=0.1,
                dashscope_api_key=dashscope_key
            )
            print("Qwen (DashScope) initialized successfully!")
        
        else:
            try:
                self.llm = ChatOllama(
                    model="qwen2:7b",
                    temperature=0.1,
                    base_url="http://localhost:11434"
                )
                print("Qwen (Ollama) initialized successfully!")
            except Exception as e:
                print(f"Warning: Could not connect to Ollama: {e}")
                print("   Please ensure Ollama is running and Qwen model is installed:")
                print("   Run: ollama pull qwen2:7b")
                raise
        
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(FraudAnalysisState)
        
        workflow.add_node("analyze_transaction", self._analyze_transaction)
        workflow.add_node("assess_risk", self._assess_risk)
        workflow.add_node("generate_explanation", self._generate_explanation)
        workflow.add_node("create_recommendations", self._create_recommendations)
        workflow.add_node("investigation_plan", self._create_investigation_plan)
        
        workflow.add_edge("analyze_transaction", "assess_risk")
        workflow.add_edge("assess_risk", "generate_explanation")
        workflow.add_edge("generate_explanation", "create_recommendations")
        workflow.add_edge("create_recommendations", "investigation_plan")
        workflow.add_edge("investigation_plan", END)
        
        workflow.set_entry_point("analyze_transaction")
        
        return workflow.compile()
    
    def _analyze_transaction(self, state: FraudAnalysisState) -> FraudAnalysisState:
        transaction_data = state["transaction_data"]
        
        df = pd.DataFrame([transaction_data])
        
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df[self.feature_names]
        
        transaction_scaled = self.scaler.transform(df)
        fraud_probability = self.model.predict_proba(transaction_scaled)[:, 1][0]
        fraud_prediction = self.model.predict(transaction_scaled)[0]
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
        else:
            feature_importance = {}
        
        state["fraud_prediction"] = {
            "fraud_probability": fraud_probability,
            "is_fraud": bool(fraud_prediction),
            "confidence": max(fraud_probability, 1 - fraud_probability),
            "feature_importance": feature_importance
        }
        
        return state
    
    def _assess_risk(self, state: FraudAnalysisState) -> FraudAnalysisState:
        fraud_prob = state["fraud_prediction"]["fraud_probability"]
        transaction_data = state["transaction_data"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fraud risk assessment expert. Analyze the transaction data and fraud probability to determine the risk level.
            
            Risk Levels:
            - LOW: Probability < 0.1, minimal risk
            - MEDIUM: Probability 0.1-0.5, moderate risk, needs monitoring
            - HIGH: Probability 0.5-0.8, significant risk, requires investigation
            - CRITICAL: Probability > 0.8, immediate action required
            
            Provide a detailed risk assessment considering:
            1. Transaction amount and timing
            2. Fraud probability score
            3. Historical patterns
            4. Business context"""),
            ("human", """Transaction Details:
            Amount: ${amount}
            Time: {time} seconds
            Fraud Probability: {fraud_prob:.4f}
            
            Please provide a comprehensive risk assessment.""")
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "amount": transaction_data.get("Amount", 0),
            "time": transaction_data.get("Time", 0),
            "fraud_prob": fraud_prob
        })
        
        state["risk_assessment"] = response.content
        
        return state
    
    def _generate_explanation(self, state: FraudAnalysisState) -> FraudAnalysisState:
        fraud_pred = state["fraud_prediction"]
        transaction_data = state["transaction_data"]
        
        feature_importance = fraud_pred["feature_importance"]
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fraud detection expert explaining why a transaction was flagged or not flagged as fraudulent.
            
            Provide a clear, non-technical explanation that includes:
            1. The main reasons for the fraud probability score
            2. Which features contributed most to the decision
            3. What patterns the model detected
            4. Context about why this matters for fraud detection
            
            Use simple language that business users can understand."""),
            ("human", """Transaction Analysis:
            Amount: ${amount}
            Time: {time} seconds
            Fraud Probability: {fraud_prob:.4f}
            Is Fraud: {is_fraud}
            
            Top Contributing Features:
            {top_features}
            
            Please explain why this transaction received this fraud score.""")
        ])
        
        chain = prompt | self.llm
        
        top_features_str = "\n".join([f"- {feat}: {imp:.4f}" for feat, imp in top_features])
        
        response = chain.invoke({
            "amount": transaction_data.get("Amount", 0),
            "time": transaction_data.get("Time", 0),
            "fraud_prob": fraud_pred["fraud_probability"],
            "is_fraud": fraud_pred["is_fraud"],
            "top_features": top_features_str
        })
        
        state["explanation"] = response.content
        
        return state
    
    def _create_recommendations(self, state: FraudAnalysisState) -> FraudAnalysisState:
        fraud_pred = state["fraud_prediction"]
        risk_assessment = state["risk_assessment"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fraud prevention expert providing actionable recommendations based on transaction analysis.
            
            Provide specific, actionable recommendations for:
            1. Immediate actions to take
            2. Monitoring and verification steps
            3. Customer communication if needed
            4. System improvements
            5. Follow-up actions
            
            Consider the risk level and fraud probability in your recommendations."""),
            ("human", """Fraud Analysis Results:
            Fraud Probability: {fraud_prob:.4f}
            Risk Level: {risk_level}
            
            Risk Assessment: {risk_assessment}
            
            Please provide specific recommendations for this transaction.""")
        ])
        
        chain = prompt | self.llm
        
        fraud_prob = fraud_pred["fraud_probability"]
        if fraud_prob < 0.1:
            risk_level = "LOW"
        elif fraud_prob < 0.5:
            risk_level = "MEDIUM"
        elif fraud_prob < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        response = chain.invoke({
            "fraud_prob": fraud_prob,
            "risk_level": risk_level,
            "risk_assessment": risk_assessment
        })
        
        recommendations = response.content.split('\n')
        recommendations = [rec.strip() for rec in recommendations if rec.strip()]
        
        state["recommendations"] = recommendations
        
        return state
    
    def _create_investigation_plan(self, state: FraudAnalysisState) -> FraudAnalysisState:
        fraud_pred = state["fraud_prediction"]
        
        if fraud_pred["fraud_probability"] < 0.5:
            state["investigation_steps"] = ["No investigation required - low risk transaction"]
            return state
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fraud investigator creating a detailed investigation plan for suspicious transactions.
            
            Create a step-by-step investigation plan including:
            1. Data collection steps
            2. Verification procedures
            3. Customer contact protocols
            4. Documentation requirements
            5. Escalation procedures
            6. Timeline for investigation
            
            Make the plan practical and actionable for fraud analysts."""),
            ("human", """Suspicious Transaction Details:
            Fraud Probability: {fraud_prob:.4f}
            Risk Level: {risk_level}
            
            Create a detailed investigation plan for this transaction.""")
        ])
        
        chain = prompt | self.llm
        
        fraud_prob = fraud_pred["fraud_probability"]
        if fraud_prob < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        response = chain.invoke({
            "fraud_prob": fraud_prob,
            "risk_level": risk_level
        })
        
        investigation_steps = response.content.split('\n')
        investigation_steps = [step.strip() for step in investigation_steps if step.strip()]
        
        state["investigation_steps"] = investigation_steps
        
        return state
    
    def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        initial_state = FraudAnalysisState(
            transaction_data=transaction_data,
            fraud_prediction={},
            risk_assessment="",
            explanation="",
            recommendations=[],
            investigation_steps=[],
            messages=[]
        )
        
        result = self.workflow.invoke(initial_state)
        
        return {
            "fraud_prediction": result["fraud_prediction"],
            "risk_assessment": result["risk_assessment"],
            "explanation": result["explanation"],
            "recommendations": result["recommendations"],
            "investigation_steps": result["investigation_steps"]
        }

def main():
    print("=== LangGraph Fraud Detection Workflow Demo ===\n")
    
    fraud_workflow = FraudDetectionWorkflow()
    
    df = pd.read_csv('creditcard.csv')
    
    print("Analyzing Normal Transaction:")
    normal_transaction = df[df['Class'] == 0].iloc[0].drop('Class').to_dict()
    
    result = fraud_workflow.analyze_transaction(normal_transaction)
    
    print(f"Fraud Probability: {result['fraud_prediction']['fraud_probability']:.4f}")
    print(f"Risk Assessment: {result['risk_assessment']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Recommendations: {result['recommendations'][:3]}")
    
    print("\n" + "="*80 + "\n")
    
    print("Analyzing Fraud Transaction:")
    fraud_transaction = df[df['Class'] == 1].iloc[0].drop('Class').to_dict()
    
    result = fraud_workflow.analyze_transaction(fraud_transaction)
    
    print(f"Fraud Probability: {result['fraud_prediction']['fraud_probability']:.4f}")
    print(f"Risk Assessment: {result['risk_assessment']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Recommendations: {result['recommendations'][:3]}")
    print(f"Investigation Steps: {result['investigation_steps'][:3]}")

if __name__ == "__main__":
    main()
