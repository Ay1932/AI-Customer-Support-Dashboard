#!/usr/bin/env python3
"""
Demo script showing how to use all advanced features of the AI Customer Support Dashboard
"""

import streamlit as st
import pandas as pd
import requests
import time
import subprocess
import sys
from pathlib import Path

def run_demo():
    """Run a comprehensive demo of all dashboard features"""

    st.title("🚀 AI Customer Support Dashboard - Feature Demo")
    st.markdown("---")

    st.header("🎯 Demo Overview")
    st.write("""
    This demo will showcase all the advanced features integrated into your dashboard:

    1. **Advanced Model Training** - Hyperparameter tuning and model comparison
    2. **Model Explainability** - SHAP and LIME explanations
    3. **REST API Testing** - Live API endpoint testing
    4. **Model Monitoring** - Performance tracking and drift detection
    5. **Testing Suite** - Automated quality assurance

    Let's explore each feature!
    """)

    # Feature showcase
    tabs = st.tabs([
        "🤖 Advanced Training",
        "🔍 Explainability",
        "🌐 API Testing",
        "📊 Monitoring",
        "🧪 Testing"
    ])

    with tabs[0]:
        st.header("🤖 Advanced Model Training Demo")

        st.write("**What it does:** Trains multiple ML models with hyperparameter optimization")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Available Models")
            models = [
                "🔍 Random Forest",
                "🚀 Gradient Boosting",
                "🎯 SVM",
                "📈 Logistic Regression",
                "🧠 Naive Bayes",
                "🎪 Ensemble (Voting)"
            ]
            for model in models:
                st.write(f"• {model}")

        with col2:
            st.subheader("Features")
            features = [
                "Grid Search CV",
                "Cross-validation",
                "Performance comparison",
                "Best model selection",
                "Hyperparameter tuning"
            ]
            for feature in features:
                st.success(f"✓ {feature}")

        st.info("💡 **Try it:** Go to '🤖 Advanced Training' page and click '🚀 Start Advanced Training'")

    with tabs[1]:
        st.header("🔍 Model Explainability Demo")

        st.write("**What it does:** Explains model predictions using advanced techniques")

        # Sample explanation
        st.subheader("Sample Prediction Explanation")

        sample_ticket = "My login is not working and I can't access my account dashboard"
        st.write(f"**Sample Ticket:** {sample_ticket}")

        # Mock explanation (in real app this would be dynamic)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Category", "Technical Issue")
            st.metric("Predicted Priority", "High")

        with col2:
            st.metric("Category Confidence", "89%")
            st.metric("Priority Confidence", "92%")

        st.subheader("Key Contributing Words")
        words = ["login", "access", "account", "dashboard", "working"]
        for word in words:
            st.write(f"• **{word}** - High importance for technical category")

        st.info("💡 **Try it:** Go to '🔍 Model Explainability' page to explore SHAP/LIME explanations")

    with tabs[2]:
        st.header("🌐 REST API Testing Demo")

        st.write("**What it does:** Provides programmatic access to model predictions")

        # API endpoints
        st.subheader("Available Endpoints")

        endpoints = {
            "GET /health": "Check API and model status",
            "GET /docs": "Interactive API documentation",
            "POST /predict": "Get predictions for ticket data",
            "GET /categories": "List available categories",
            "GET /priorities": "List available priorities"
        }

        for endpoint, desc in endpoints.items():
            st.code(f"{endpoint}", language="http")
            st.write(f"*{desc}*")
            st.write("---")

        # Sample API call
        st.subheader("Sample API Usage")

        st.code("""
# Python example
import requests

url = "http://localhost:8000/predict"
ticket = {
    "subject": "Payment failed",
    "description": "My card was declined during checkout"
}

response = requests.post(url, json=ticket)
result = response.json()
print(f"Category: {result['category']}")
print(f"Priority: {result['priority']}")
        """, language="python")

        st.info("💡 **Try it:** Start the API with `make run-api` then test in '🌐 API Testing' page")

    with tabs[3]:
        st.header("📊 Model Monitoring Demo")

        st.write("**What it does:** Tracks model performance and detects drift over time")

        # Mock monitoring data
        st.subheader("Performance Tracking")

        # Create sample monitoring data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='W')
        cat_accuracy = [0.85, 0.87, 0.86, 0.88, 0.85, 0.84, 0.86, 0.87, 0.85, 0.86]
        pri_accuracy = [0.82, 0.84, 0.83, 0.85, 0.82, 0.81, 0.83, 0.84, 0.82, 0.83]

        monitor_df = pd.DataFrame({
            'Date': dates,
            'Category_Accuracy': cat_accuracy,
            'Priority_Accuracy': pri_accuracy
        })

        st.line_chart(monitor_df.set_index('Date'))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Category Accuracy", "86.0%")
        with col2:
            st.metric("Current Priority Accuracy", "83.0%")
        with col3:
            st.metric("Evaluations", "10")

        st.subheader("Drift Detection")
        st.success("✅ No significant model drift detected")
        st.info("Drift detection compares recent performance against baseline")

        st.info("💡 **Try it:** Go to '📊 Model Monitoring' page to run evaluations and check for drift")

    with tabs[4]:
        st.header("🧪 Testing Suite Demo")

        st.write("**What it does:** Runs automated tests to ensure code quality")

        st.subheader("Test Categories")

        test_types = {
            "Unit Tests": "Test individual functions",
            "Integration Tests": "Test model pipelines",
            "Data Processing Tests": "Test text preprocessing",
            "Model Loading Tests": "Test model loading",
            "API Tests": "Test API endpoints",
            "Performance Tests": "Test prediction speed"
        }

        for test_name, desc in test_types.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{test_name}**")
                st.write(f"*{desc}*")
            with col2:
                coverage = {"Unit Tests": 85, "Integration Tests": 92, "Data Processing Tests": 95,
                           "Model Loading Tests": 88, "API Tests": 78, "Performance Tests": 82}
                st.metric("Coverage", f"{coverage[test_name]}%")

        st.subheader("Sample Test Results")
        st.success("✅ Unit Tests: 15/15 passed")
        st.success("✅ Integration Tests: 8/8 passed")
        st.success("✅ Data Processing Tests: 5/5 passed")

        st.info("💡 **Try it:** Go to '🧪 Testing Suite' page to run tests and view coverage reports")

    # Quick start guide
    st.markdown("---")
    st.header("🚀 Quick Start Guide")

    st.subheader("1. Install Dependencies")
    st.code("pip install -r requirements.txt", language="bash")

    st.subheader("2. Train Basic Models")
    st.code("python data_preprocessing.py && python train_models.py", language="bash")

    st.subheader("3. Run Dashboard")
    st.code("streamlit run app.py", language="bash")

    st.subheader("4. Explore Advanced Features")
    st.write("Use the sidebar to navigate to advanced pages!")

    st.success("🎉 **Your dashboard now includes enterprise-grade ML features!**")

if __name__ == "__main__":
    run_demo()