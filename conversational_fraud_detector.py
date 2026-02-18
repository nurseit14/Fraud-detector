#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

class ChatState(TypedDict):
    messages: List[Any]
    current_transaction: Dict[str, Any]
    fraud_analysis: Dict[str, Any]
    conversation_context: Dict[str, Any]

class ConversationalFraudDetector:
    
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
        
        self.workflow = self._create_conversation_workflow()
    
    def _create_conversation_workflow(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        
        workflow.add_node("understand_intent", self._understand_intent)
        workflow.add_node("extract_transaction", self._extract_transaction_data)
        workflow.add_node("analyze_fraud", self._analyze_fraud)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("handle_general_query", self._handle_general_query)
        
        workflow.add_conditional_edges(
            "understand_intent",
            self._route_intent,
            {
                "analyze_transaction": "extract_transaction",
                "general_query": "handle_general_query",
                "end_conversation": END
            }
        )
        
        workflow.add_edge("extract_transaction", "analyze_fraud")
        workflow.add_edge("analyze_fraud", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_general_query", END)
        
        workflow.set_entry_point("understand_intent")
        
        return workflow.compile()
    
    def _understand_intent(self, state: ChatState) -> ChatState:
        last_message = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a fraud detection system. 
            Classify the user's intent into one of these categories:
            
            1. "analyze_transaction" - User wants to analyze a specific transaction for fraud
            2. "general_query" - User has general questions about fraud detection
            3. "end_conversation" - User wants to end the conversation
            
            Respond with only the intent category."""),
            ("human", "User message: {message}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"message": last_message})
        
        intent = response.content.strip().lower()
        state["conversation_context"] = {"intent": intent}
        
        return state
    
    def _route_intent(self, state: ChatState) -> str:
        intent = state["conversation_context"]["intent"]
        
        if "analyze_transaction" in intent:
            return "analyze_transaction"
        elif "general_query" in intent:
            return "general_query"
        elif "end_conversation" in intent:
            return "end_conversation"
        else:
            return "general_query"
    
    def _extract_transaction_data(self, state: ChatState) -> ChatState:
        user_message = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data extraction expert. Extract transaction information from user messages.
            
            Extract the following fields if mentioned:
            - Amount: Transaction amount in dollars
            - Time: Time in seconds (or convert from other formats)
            - V1-V28: Any mentioned feature values
            
            If a field is not mentioned, set it to 0.
            Return the data as a JSON object with these exact field names:
            Amount, Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28
            
            Example response:
            {{"Amount": 100.0, "Time": 1000, "V1": 1.2, "V2": -0.5, ...}}"""),
            ("human", "Extract transaction data from: {message}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"message": user_message})
        
        try:
            import json
            transaction_data = json.loads(response.content)
        except:
            transaction_data = {f"V{i}": 0 for i in range(1, 29)}
            transaction_data["Amount"] = 0
            transaction_data["Time"] = 0
        
        state["current_transaction"] = transaction_data
        
        return state
    
    def _analyze_fraud(self, state: ChatState) -> ChatState:
        transaction_data = state["current_transaction"]
        
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
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = []
        
        state["fraud_analysis"] = {
            "fraud_probability": fraud_probability,
            "is_fraud": bool(fraud_prediction),
            "confidence": max(fraud_probability, 1 - fraud_probability),
            "top_features": top_features
        }
        
        return state
    
    def _generate_response(self, state: ChatState) -> ChatState:
        fraud_analysis = state["fraud_analysis"]
        transaction_data = state["current_transaction"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful fraud detection assistant. Provide a conversational response about the fraud analysis.
            
            Include:
            1. The fraud probability and what it means
            2. Whether the transaction is flagged as fraud
            3. The risk level (LOW/MEDIUM/HIGH/CRITICAL)
            4. Key factors that influenced the decision
            5. What the user should do next
            
            Be conversational, helpful, and easy to understand."""),
            ("human", """Transaction Analysis Results:
            Amount: ${amount}
            Time: {time} seconds
            Fraud Probability: {fraud_prob:.4f}
            Is Fraud: {is_fraud}
            Top Contributing Features: {top_features}
            
            Please provide a helpful response about this analysis.""")
        ])
        
        chain = prompt | self.llm
        
        top_features_str = ", ".join([f"{feat} ({imp:.3f})" for feat, imp in fraud_analysis["top_features"]])
        
        response = chain.invoke({
            "amount": transaction_data.get("Amount", 0),
            "time": transaction_data.get("Time", 0),
            "fraud_prob": fraud_analysis["fraud_probability"],
            "is_fraud": fraud_analysis["is_fraud"],
            "top_features": top_features_str
        })
        
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    def _handle_general_query(self, state: ChatState) -> ChatState:
        user_message = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fraud detection expert assistant. Answer general questions about fraud detection, 
            the system, or provide guidance on how to use it.
            
            You can help with:
            - Explaining how fraud detection works
            - Describing the features used in analysis
            - Providing guidance on transaction analysis
            - Explaining risk levels and recommendations
            
            Be helpful, accurate, and conversational."""),
            ("human", "User question: {message}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"message": user_message})
        
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    def chat(self, user_input: str) -> str:
        if not hasattr(self, '_current_state'):
            self._current_state = ChatState(
                messages=[SystemMessage(content="You are a fraud detection assistant. Help users analyze transactions for fraud.")],
                current_transaction={},
                fraud_analysis={},
                conversation_context={}
            )
        
        self._current_state["messages"].append(HumanMessage(content=user_input))
        
        result = self.workflow.invoke(self._current_state)
        
        self._current_state = result
        
        return result["messages"][-1].content

def interactive_chat():
    print("Fraud Detection Assistant")
    print("=" * 50)
    print("I can help you analyze transactions for fraud!")
    print("Try saying things like:")
    print("- 'Analyze a transaction with amount $100'")
    print("- 'How does fraud detection work?'")
    print("- 'What features are most important?'")
    print("- 'quit' to exit")
    print("=" * 50)
    
    detector = ConversationalFraudDetector()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Assistant: Goodbye! Stay safe from fraud!")
                break
            
            if not user_input:
                continue
            
            print("Assistant: ", end="")
            response = detector.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\nAssistant: Goodbye! Stay safe from fraud!")
            break
        except Exception as e:
            print(f"Assistant: Sorry, I encountered an error: {e}")

def demo_conversation():
    print("=== Conversational Fraud Detection Demo ===\n")
    
    detector = ConversationalFraudDetector()
    
    demo_inputs = [
        "Hello! Can you help me analyze a transaction?",
        "I have a transaction with amount $150 and time 1000 seconds",
        "How does your fraud detection work?",
        "What are the most important features for fraud detection?",
        "Analyze a suspicious transaction: amount $5000, time 50000 seconds"
    ]
    
    for user_input in demo_inputs:
        print(f"User: {user_input}")
        response = detector.chat(user_input)
        print(f"Assistant: {response}\n")
        print("-" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_conversation()
    else:
        interactive_chat()

