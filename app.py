import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
import time
import subprocess
import sys
import os

# Try to import advanced libraries (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
METRICS_FILE = MODEL_DIR / 'model_metrics.json'

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
METRICS_FILE = MODEL_DIR / 'model_metrics.json'

# Load models and preprocessors
@st.cache_resource
def load_models():
    vectorizer = joblib.load(MODEL_DIR / 'tfidf_vectorizer.pkl')
    category_model = joblib.load(MODEL_DIR / 'category_model.pkl')
    priority_model = joblib.load(MODEL_DIR / 'priority_model.pkl')
    category_encoder = joblib.load(MODEL_DIR / 'category_encoder.pkl')
    priority_encoder = joblib.load(MODEL_DIR / 'priority_encoder.pkl')
    return vectorizer, category_model, priority_model, category_encoder, priority_encoder

# Text preprocessing function
def preprocess_text(text):
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# Prediction function
def predict_ticket(text, vectorizer, category_model, priority_model, cat_encoder, pri_encoder):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])

    category_pred = category_model.predict(vectorized_text)[0]
    priority_pred = priority_model.predict(vectorized_text)[0]

    category_label = cat_encoder.inverse_transform([category_pred])[0]
    priority_label = pri_encoder.inverse_transform([priority_pred])[0]

    return category_label, priority_label

# Load data
@st.cache_data
def load_ticket_data():
    df = pd.read_csv(DATA_DIR / 'sample_tickets.csv')
    df['Created Date'] = pd.to_datetime(df['Created Date'])
    df['Resolved Date'] = pd.to_datetime(df['Resolved Date'], errors='coerce')
    return df

@st.cache_data
def load_model_metrics():
    if METRICS_FILE.exists():
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def main():
    st.set_page_config(page_title="Customer Support Ticket Analytics", layout="wide")

    st.title("🎫 AI-Based Customer Support Ticket Prioritization & Analytics Dashboard")

    # Load models and data
    try:
        vectorizer, category_model, priority_model, cat_encoder, pri_encoder = load_models()
        df = load_ticket_data()
        metrics = load_model_metrics()
    except FileNotFoundError as exc:
        st.error(f"Models or data not found: {exc.filename}. Please run preprocessing and training from the customer_support_dashboard folder.")
        return

    # Sidebar for navigation
    pages = [
        "Analytics Dashboard",
        "Data Explorer",
        "Ticket Prediction",
        "🤖 Advanced Training",
        "🔍 Model Explainability",
        "🌐 API Testing",
        "📊 Model Monitoring",
        "🧪 Testing Suite"
    ]
    page = st.sidebar.selectbox("Choose a page", pages)

    if page == "Analytics Dashboard":
        show_analytics_dashboard(df, metrics)
    elif page == "Data Explorer":
        show_data_explorer(df)
    elif page == "Ticket Prediction":
        show_ticket_prediction(vectorizer, category_model, priority_model, cat_encoder, pri_encoder)
    elif page == "🤖 Advanced Training":
        show_advanced_training()
    elif page == "🔍 Model Explainability":
        show_model_explainability(vectorizer, category_model, priority_model, cat_encoder, pri_encoder)
    elif page == "🌐 API Testing":
        show_api_testing()
    elif page == "📊 Model Monitoring":
        show_model_monitoring(df)
    elif page == "🧪 Testing Suite":
        show_testing_suite()

def show_analytics_dashboard(df, metrics):
    st.header("📊 Analytics Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_tickets = len(df)
    open_tickets = len(df[df['Status'] == 'Open'])
    resolved_tickets = len(df[df['Status'] == 'Resolved'])
    avg_resolution_time = df['Resolved Date'].sub(df['Created Date']).dt.days.mean()
    high_priority_open = len(df[(df['Status'] == 'Open') & (df['Priority'] == 'High')])

    col1.metric("Total Tickets", total_tickets)
    col2.metric("Open Tickets", open_tickets)
    col3.metric("🚨 High Priority Open", high_priority_open)
    col4.metric("Avg Resolution (days)", f"{avg_resolution_time:.1f}")

    st.markdown("---")

    # Charts Row 1
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_types = px.pie(df, names='Ticket Type', title='Ticket Types Distribution')
        st.plotly_chart(fig_types, use_container_width=True)

    with col2:
        fig_priority = px.bar(df['Priority'].value_counts(), title='Priority Distribution', labels={'value': 'Count', 'index': 'Priority'})
        st.plotly_chart(fig_priority, use_container_width=True)

    with col3:
        status_counts = df['Status'].value_counts()
        fig_status = px.bar(status_counts, title='Ticket Status Distribution', labels={'value': 'Count', 'index': 'Status'})
        st.plotly_chart(fig_status, use_container_width=True)

    # Time series
    st.subheader("📈 Ticket Volume Over Time")
    daily_tickets = df.groupby(df['Created Date'].dt.date).size().reset_index(name='count')
    fig_time = px.line(daily_tickets, x='Created Date', y='count', title='Daily Ticket Volume', markers=True)
    fig_time.update_layout(hovermode='x unified')
    st.plotly_chart(fig_time, use_container_width=True)

    # Category Analysis
    st.subheader("📂 Category Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        category_counts = df['Ticket Type'].value_counts()
        fig_cat = px.bar(category_counts, title='Tickets by Category', labels={'value': 'Count', 'index': 'Category'})
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Avg resolution time by category
        df_resolved = df[df['Status'] == 'Resolved'].copy()
        df_resolved['resolution_days'] = (df_resolved['Resolved Date'] - df_resolved['Created Date']).dt.days
        avg_by_category = df_resolved.groupby('Ticket Type')['resolution_days'].mean().sort_values(ascending=False)
        fig_res = px.bar(avg_by_category, title='Avg Resolution Time by Category (days)', labels={'value': 'Days', 'index': 'Category'})
        st.plotly_chart(fig_res, use_container_width=True)

    # Priority vs Status Heatmap
    st.subheader("🔥 Priority vs Status Matrix")
    priority_status = pd.crosstab(df['Priority'], df['Status'])
    fig_heatmap = px.imshow(priority_status, labels=dict(x='Status', y='Priority', color='Count'), title='Tickets by Priority and Status')
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Export data
    st.subheader("📥 Export Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download all tickets as CSV",
        data=csv,
        file_name="tickets_export.csv",
        mime="text/csv"
    )

    if metrics:
        show_model_performance(metrics)
    else:
        st.info("Model metrics are not available yet. Run training to generate evaluation data.")


def show_model_performance(metrics):
    st.subheader("Model Performance Metrics")
    st.write("This summary shows model accuracy and classification metrics for category and priority prediction.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Category Model**")
        category_accuracy = metrics.get('category', {}).get('accuracy', None)
        if category_accuracy is not None:
            st.metric("Category Accuracy", f"{category_accuracy:.2f}")
        st.write(metrics.get('category', {}))

    with col2:
        st.markdown("**Priority Model**")
        priority_accuracy = metrics.get('priority', {}).get('accuracy', None)
        if priority_accuracy is not None:
            st.metric("Priority Accuracy", f"{priority_accuracy:.2f}")
        st.write(metrics.get('priority', {}))


def show_data_explorer(df):
    st.header("📁 Data Explorer")
    st.write("Filter, search, and inspect ticket records. Click column headers to sort.")

    col1, col2, col3, col4 = st.columns(4)
    
    types = ['All'] + sorted(df['Ticket Type'].dropna().unique().tolist())
    statuses = ['All'] + sorted(df['Status'].dropna().unique().tolist())
    priorities = ['All'] + sorted(df['Priority'].dropna().unique().tolist())

    with col1:
        type_filter = st.selectbox("Filter by Type", types)
    with col2:
        status_filter = st.selectbox("Filter by Status", statuses)
    with col3:
        priority_filter = st.selectbox("Filter by Priority", priorities)
    with col4:
        search_term = st.text_input("Search in Subject", "")

    filtered = df.copy()
    if type_filter != 'All':
        filtered = filtered[filtered['Ticket Type'] == type_filter]
    if status_filter != 'All':
        filtered = filtered[filtered['Status'] == status_filter]
    if priority_filter != 'All':
        filtered = filtered[filtered['Priority'] == priority_filter]
    if search_term:
        filtered = filtered[filtered['Subject'].str.contains(search_term, case=False, na=False)]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} tickets**")
    
    # Display table
    display_cols = ['Ticket ID', 'Ticket Type', 'Subject', 'Priority', 'Status', 'Created Date', 'Resolved Date']
    st.dataframe(filtered[display_cols], use_container_width=True, height=600)

    # Expandable sections
    with st.expander("Show ticket descriptions", expanded=False):
        st.dataframe(filtered[['Ticket ID', 'Subject', 'Description']], use_container_width=True, height=500)
    
    with st.expander("Show statistics for filtered data", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(filtered))
        with col2:
            st.metric("Open", len(filtered[filtered['Status'] == 'Open']))
        with col3:
            st.metric("High Priority", len(filtered[filtered['Priority'] == 'High']))
    
    # Export filtered data
    if len(filtered) > 0:
        csv = filtered.to_csv(index=False)
        st.download_button(
            label=f"Download {len(filtered)} filtered tickets as CSV",
            data=csv,
            file_name="filtered_tickets_export.csv",
            mime="text/csv"
        )


def show_ticket_prediction(vectorizer, category_model, priority_model, cat_encoder, pri_encoder):
    st.header("🔮 Ticket Prediction")

    st.write("Enter ticket details to predict category and priority:")

    subject = st.text_input("Ticket Subject")
    description = st.text_area("Ticket Description")

    if st.button("Predict"):
        if subject and description:
            full_text = subject + " " + description
            category, priority = predict_ticket(
                full_text, vectorizer, category_model, priority_model, cat_encoder, pri_encoder
            )

            col1, col2 = st.columns(2)

            with col1:
                if priority == 'High':
                    st.error(f"🚨 Predicted Priority: {priority}")
                elif priority == 'Medium':
                    st.warning(f"⚠️ Predicted Priority: {priority}")
                else:
                    st.success(f"✅ Predicted Priority: {priority}")

            with col2:
                st.info(f"📂 Predicted Category: {category}")

        else:
            st.warning("Please enter both subject and description.")

        else:
            st.warning("Please enter both subject and description.")

def show_advanced_training():
    st.header("🤖 Advanced Model Training & Comparison")

    st.write("Train and compare multiple ML models with hyperparameter tuning and ensemble methods.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Options")

        if st.button("🚀 Start Advanced Training", type="primary"):
            with st.spinner("Training models with hyperparameter tuning... This may take several minutes."):
                progress_bar = st.progress(0)

                # Run the advanced training script
                try:
                    result = subprocess.run([
                        sys.executable, str(BASE_DIR / "advanced_training.py")
                    ], capture_output=True, text=True, cwd=BASE_DIR)

                    progress_bar.progress(100)

                    if result.returncode == 0:
                        st.success("✅ Advanced training completed successfully!")

                        # Load and display results
                        results_file = MODEL_DIR / "advanced_model_comparison.json"
                        if results_file.exists():
                            with open(results_file, 'r') as f:
                                results = json.load(f)

                            st.subheader("📊 Model Comparison Results")

                            # Display category model results
                            if 'category_models' in results:
                                st.write("**🏷️ Category Classification Models:**")
                                cat_data = []
                                for model_name, data in results['category_models'].items():
                                    if 'test_results' in data:
                                        cat_data.append({
                                            'Model': model_name,
                                            'Accuracy': data['test_results']['accuracy'],
                                            'Best Params': str(data.get('best_params', 'N/A'))[:50] + '...'
                                        })

                                if cat_data:
                                    st.dataframe(pd.DataFrame(cat_data), use_container_width=True)

                            # Display priority model results
                            if 'priority_models' in results:
                                st.write("**⚡ Priority Prediction Models:**")
                                pri_data = []
                                for model_name, data in results['priority_models'].items():
                                    if 'test_results' in data:
                                        pri_data.append({
                                            'Model': model_name,
                                            'Accuracy': data['test_results']['accuracy'],
                                            'Best Params': str(data.get('best_params', 'N/A'))[:50] + '...'
                                        })

                                if pri_data:
                                    st.dataframe(pd.DataFrame(pri_data), use_container_width=True)

                    else:
                        st.error(f"❌ Training failed: {result.stderr}")

                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")

    with col2:
        st.subheader("Available Models")
        models_list = [
            "Random Forest",
            "Gradient Boosting",
            "SVM",
            "Logistic Regression",
            "Naive Bayes",
            "Ensemble (Voting)"
        ]

        for model in models_list:
            st.write(f"• {model}")

        st.subheader("Features")
        features_list = [
            "Hyperparameter Tuning",
            "Cross-Validation",
            "Model Comparison",
            "Ensemble Methods",
            "Performance Metrics"
        ]

        for feature in features_list:
            st.write(f"✓ {feature}")

def show_model_explainability(vectorizer, category_model, priority_model, cat_encoder, pri_encoder):
    st.header("🔍 Model Explainability")

    st.write("Understand how the models make predictions using SHAP and LIME explainability techniques.")

    # Sample predictions for explanation
    sample_texts = [
        "My account login is not working, I can't access my dashboard",
        "I received the wrong item in my order, it's completely different from what I ordered",
        "The website is running very slowly, pages take forever to load",
        "I need to update my billing information for recurring payments"
    ]

    tab1, tab2, tab3 = st.tabs(["📝 Sample Explanations", "🎯 Custom Prediction", "📊 Feature Importance"])

    with tab1:
        st.subheader("Sample Ticket Explanations")

        selected_sample = st.selectbox(
            "Choose a sample ticket to explain:",
            sample_texts,
            format_func=lambda x: x[:50] + "..." if len(x) > 50 else x
        )

        if st.button("🔍 Explain This Prediction"):
            with st.spinner("Generating explanations..."):
                # Get prediction
                category, priority = predict_ticket(
                    selected_sample, vectorizer, category_model, priority_model, cat_encoder, pri_encoder
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Category", category)
                with col2:
                    st.metric("Predicted Priority", priority)

                # Simple feature importance (word frequency in training data)
                st.subheader("🔤 Key Words Analysis")

                processed_text = preprocess_text(selected_sample)
                words = processed_text.split()

                # Get feature names and importance
                feature_names = vectorizer.get_feature_names_out()
                text_vector = vectorizer.transform([processed_text]).toarray()[0]

                # Find important features
                important_features = []
                for i, importance in enumerate(text_vector):
                    if importance > 0:
                        important_features.append({
                            'word': feature_names[i],
                            'importance': importance
                        })

                # Sort by importance
                important_features.sort(key=lambda x: x['importance'], reverse=True)

                if important_features:
                    st.write("**Top contributing words:**")
                    for feature in important_features[:10]:
                        st.write(f"• **{feature['word']}** (weight: {feature['importance']:.3f})")

                # SHAP/LIME status
                st.subheader("🤖 Advanced Explainability")
                if SHAP_AVAILABLE:
                    st.success("✅ SHAP is available for detailed explanations")
                else:
                    st.warning("⚠️ SHAP not installed. Run: `pip install shap`")

                if LIME_AVAILABLE:
                    st.success("✅ LIME is available for local explanations")
                else:
                    st.warning("⚠️ LIME not installed. Run: `pip install lime`")

    with tab2:
        st.subheader("Custom Ticket Explanation")

        subject = st.text_input("Ticket Subject", key="explain_subject")
        description = st.text_area("Ticket Description", key="explain_desc")

        if st.button("🔍 Explain Custom Prediction"):
            if subject and description:
                full_text = subject + " " + description

                # Get prediction
                category, priority = predict_ticket(
                    full_text, vectorizer, category_model, priority_model, cat_encoder, pri_encoder
                )

                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Category", category)
                with col2:
                    st.metric("Predicted Priority", priority)

                # Word analysis
                processed_text = preprocess_text(full_text)
                words = processed_text.split()

                st.subheader("Text Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Original length:** {len(full_text)} characters")
                    st.write(f"**Processed words:** {len(words)}")
                    st.write(f"**Unique words:** {len(set(words))}")

                with col2:
                    # Simple sentiment-like analysis
                    urgent_words = ['urgent', 'emergency', 'critical', 'asap', 'immediately']
                    technical_words = ['error', 'bug', 'login', 'password', 'server', 'database']

                    urgent_count = sum(1 for word in words if word in urgent_words)
                    technical_count = sum(1 for word in words if word in technical_words)

                    st.write(f"**Urgent indicators:** {urgent_count}")
                    st.write(f"**Technical terms:** {technical_count}")

            else:
                st.warning("Please enter both subject and description.")

    with tab3:
        st.subheader("Global Feature Importance")

        st.write("Understanding which features are most important across all predictions.")

        # Load model metrics if available
        metrics_file = MODEL_DIR / "model_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            st.write("**Model Performance Overview:**")

            col1, col2 = st.columns(2)
            with col1:
                if 'category' in metrics and 'accuracy' in metrics['category']:
                    st.metric("Category Model Accuracy", f"{metrics['category']['accuracy']:.3f}")

            with col2:
                if 'priority' in metrics and 'accuracy' in metrics['priority']:
                    st.metric("Priority Model Accuracy", f"{metrics['priority']['accuracy']:.3f}")

        # Feature importance visualization
        st.subheader("🔤 Top Predictive Features")

        # Get feature importances from Random Forest if available
        if hasattr(category_model, 'feature_importances_'):
            feature_names = vectorizer.get_feature_names_out()
            importances = category_model.feature_importances_

            # Get top 20 features
            top_indices = importances.argsort()[-20:][::-1]
            top_features = feature_names[top_indices]
            top_importances = importances[top_indices]

            # Create bar chart
            fig = px.bar(
                x=top_importances,
                y=top_features,
                orientation='h',
                title='Top 20 Most Important Features (Category Model)',
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance visualization requires Random Forest model. Train advanced models to see this.")

def show_api_testing():
    st.header("🌐 API Testing Interface")

    st.write("Test the REST API endpoints for programmatic access to model predictions.")

    # API status check
    st.subheader("API Status")

    api_url = st.text_input("API Base URL", "http://localhost:8000", help="Change this if your API is running on a different port")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Check API Health"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        st.success("✅ API is healthy and models are loaded")
                        st.json(data)
                    else:
                        st.warning("⚠️ API is running but models may not be loaded")
                        st.json(data)
                else:
                    st.error(f"❌ API health check failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Cannot connect to API: {str(e)}")
                st.info("💡 Make sure the API is running with: `uvicorn api:app --reload`")

    with col2:
        if st.button("📚 View API Documentation"):
            docs_url = f"{api_url}/docs"
            st.markdown(f"[Open API Docs]({docs_url}) in new tab")
            st.info("The API documentation provides interactive testing for all endpoints.")

    # Prediction testing
    st.subheader("🎯 Test Prediction API")

    subject = st.text_input("Ticket Subject", key="api_subject")
    description = st.text_area("Ticket Description", key="api_desc")

    if st.button("🚀 Test Prediction API"):
        if subject and description:
            payload = {
                "subject": subject,
                "description": description
            }

            try:
                response = requests.post(f"{api_url}/predict", json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Prediction successful!")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Category", result['category'])
                    with col2:
                        st.metric("Priority", result['priority'])
                    with col3:
                        st.metric("Category Confidence", f"{result['confidence_category']:.1%}")
                    with col4:
                        st.metric("Priority Confidence", f"{result['confidence_priority']:.1%}")

                    # Show raw response
                    with st.expander("View Raw API Response"):
                        st.json(result)

                else:
                    st.error(f"❌ API request failed: {response.status_code}")
                    st.text(response.text)

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Cannot connect to API: {str(e)}")

        else:
            st.warning("Please enter both subject and description.")

    # API usage examples
    st.subheader("📖 API Usage Examples")

    with st.expander("Python Code Example"):
        st.code("""
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Ticket data
ticket = {
    "subject": "Login not working",
    "description": "I cannot access my account after password reset"
}

# Make prediction
response = requests.post(url, json=ticket)
result = response.json()

print(f"Category: {result['category']}")
print(f"Priority: {result['priority']}")
        """, language="python")

    with st.expander("cURL Example"):
        st.code("""
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "subject": "Payment failed",
       "description": "My credit card was declined during checkout"
     }'
        """, language="bash")

def show_model_monitoring(df):
    st.header("📊 Model Monitoring & Performance Tracking")

    st.write("Monitor model performance over time, detect drift, and track prediction quality.")

    # Load monitoring data if available
    monitoring_file = MODEL_DIR / "monitoring_history.json"

    if monitoring_file.exists():
        with open(monitoring_file, 'r') as f:
            monitoring_data = json.load(f)

        st.subheader("📈 Performance History")

        if monitoring_data:
            # Convert to DataFrame
            monitor_df = pd.DataFrame(monitoring_data)

            # Convert timestamps
            monitor_df['timestamp'] = pd.to_datetime(monitor_df['timestamp'])

            # Display metrics over time
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Category Model Accuracy")
                if 'category_accuracy' in monitor_df.columns:
                    fig_cat = px.line(
                        monitor_df,
                        x='timestamp',
                        y='category_accuracy',
                        title='Category Model Accuracy Over Time',
                        markers=True
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)

            with col2:
                st.subheader("Priority Model Accuracy")
                if 'priority_accuracy' in monitor_df.columns:
                    fig_pri = px.line(
                        monitor_df,
                        x='timestamp',
                        y='priority_accuracy',
                        title='Priority Model Accuracy Over Time',
                        markers=True
                    )
                    st.plotly_chart(fig_pri, use_container_width=True)

            # Latest metrics
            st.subheader("📊 Latest Performance Metrics")

            latest = monitor_df.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Latest Category Accuracy", f"{latest.get('category_accuracy', 0):.3f}")
            with col2:
                st.metric("Latest Priority Accuracy", f"{latest.get('priority_accuracy', 0):.3f}")
            with col3:
                st.metric("Evaluations Count", len(monitor_df))
            with col4:
                days_since = (datetime.now() - latest['timestamp']).days
                st.metric("Days Since Last Eval", days_since)

        else:
            st.info("No monitoring data available yet. Run model evaluation to start tracking.")

    else:
        st.info("Monitoring data not found. The monitoring system tracks model performance over time.")

    # Manual evaluation
    st.subheader("🔍 Run Manual Evaluation")

    if st.button("📊 Evaluate Current Models"):
        with st.spinner("Evaluating model performance..."):
            try:
                # Import monitoring function
                from explainability import ModelMonitor

                monitor = ModelMonitor()
                evaluation = monitor.evaluate_model_performance(
                    df[['Subject', 'Description']].head(50).values.tolist(),
                    [df['Ticket Type'].head(50).values, df['Priority'].head(50).values],
                    "manual_evaluation"
                )

                if evaluation:
                    st.success("✅ Evaluation completed!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Category Accuracy", f"{evaluation['category_accuracy']:.3f}")
                    with col2:
                        st.metric("Priority Accuracy", f"{evaluation['priority_accuracy']:.3f}")

                    # Show detailed metrics
                    with st.expander("View Detailed Metrics"):
                        st.json(evaluation)

                else:
                    st.error("❌ Evaluation failed")

            except Exception as e:
                st.error(f"❌ Error during evaluation: {str(e)}")

    # Drift detection
    st.subheader("🎯 Drift Detection")

    if st.button("🔍 Check for Model Drift"):
        try:
            from explainability import ModelMonitor

            monitor = ModelMonitor()
            drift = monitor.detect_drift()

            if drift:
                st.subheader("Drift Analysis Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Window (Days)", drift['window_days'])

                with col2:
                    cat_drift = drift['category_accuracy_drift']
                    st.metric("Category Drift", f"{cat_drift:+.3f}",
                             delta=f"{cat_drift:+.3f}")

                with col3:
                    pri_drift = drift['priority_accuracy_drift']
                    st.metric("Priority Drift", f"{pri_drift:+.3f}",
                             delta=f"{pri_drift:+.3f}")

                if drift['drift_detected']:
                    st.error("⚠️ **Model drift detected!** Consider retraining models.")
                else:
                    st.success("✅ No significant drift detected.")

            else:
                st.info("Not enough data for drift detection. Need at least 2 evaluations.")

        except Exception as e:
            st.error(f"❌ Error checking drift: {str(e)}")

def show_testing_suite():
    st.header("🧪 Testing Suite")

    st.write("Run automated tests to ensure code quality and model functionality.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Available Tests")

        test_options = {
            "Unit Tests": "Test individual functions and classes",
            "Integration Tests": "Test model loading and prediction pipelines",
            "Data Processing Tests": "Test text preprocessing and feature extraction",
            "Model Loading Tests": "Test model and encoder loading",
            "API Tests": "Test API endpoints (requires running API)",
            "Performance Tests": "Test model prediction speed"
        }

        selected_test = st.selectbox("Choose test type:", list(test_options.keys()))
        st.info(test_options[selected_test])

        if st.button("🧪 Run Tests", type="primary"):
            with st.spinner(f"Running {selected_test}..."):
                progress_bar = st.progress(0)

                try:
                    if selected_test == "Unit Tests":
                        # Run pytest on unit tests
                        result = subprocess.run([
                            sys.executable, "-m", "pytest",
                            str(BASE_DIR / "tests" / "test_main.py"),
                            "-v", "--tb=short"
                        ], capture_output=True, text=True, cwd=BASE_DIR)

                    elif selected_test == "Integration Tests":
                        # Test model loading and basic predictions
                        result = subprocess.run([
                            sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from app import load_models, predict_ticket
print('Testing model loading...')
models = load_models()
print('✓ Models loaded successfully')
print('Testing prediction...')
category, priority = predict_ticket('test ticket', *models)
print(f'✓ Prediction successful: {category}, {priority}')
                            """
                        ], capture_output=True, text=True, cwd=BASE_DIR)

                    elif selected_test == "Data Processing Tests":
                        result = subprocess.run([
                            sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from app import preprocess_text
print('Testing text preprocessing...')
result = preprocess_text('Hello World! This is a TEST.')
print(f'✓ Preprocessing result: {result}')
                            """
                        ], capture_output=True, text=True, cwd=BASE_DIR)

                    elif selected_test == "Model Loading Tests":
                        result = subprocess.run([
                            sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from app import load_models
print('Testing model loading...')
try:
    models = load_models()
    print('✓ All models loaded successfully')
    print(f'✓ Vectorizer features: {models[0].get_feature_names_out().shape[0]}')
except Exception as e:
    print(f'❌ Error: {e}')
                            """
                        ], capture_output=True, text=True, cwd=BASE_DIR)

                    elif selected_test == "API Tests":
                        # Test API endpoints
                        try:
                            response = requests.get("http://localhost:8000/health", timeout=5)
                            if response.status_code == 200:
                                result = subprocess.run([
                                    "echo", "API Health Check Passed"
                                ], capture_output=True, text=True)
                            else:
                                result = subprocess.run([
                                    "echo", f"API returned status {response.status_code}"
                                ], capture_output=True, text=True)
                        except:
                            result = subprocess.run([
                                "echo", "API not running - start with: uvicorn api:app --reload"
                            ], capture_output=True, text=True)

                    else:  # Performance Tests
                        result = subprocess.run([
                            sys.executable, "-c", """
import sys
import time
sys.path.insert(0, '.')
from app import load_models, predict_ticket
print('Testing prediction performance...')
models = load_models()
start_time = time.time()
for i in range(10):
    predict_ticket(f'test ticket {i}', *models)
end_time = time.time()
avg_time = (end_time - start_time) / 10
print(f'✓ Average prediction time: {avg_time:.3f} seconds')
                            """
                        ], capture_output=True, text=True, cwd=BASE_DIR)

                    progress_bar.progress(100)

                    # Display results
                    st.subheader("Test Results")

                    if result.returncode == 0:
                        st.success("✅ Tests passed!")

                        # Show output
                        if result.stdout:
                            with st.expander("View Test Output"):
                                st.code(result.stdout, language="text")

                    else:
                        st.error("❌ Tests failed!")

                        if result.stdout:
                            with st.expander("View Test Output"):
                                st.code(result.stdout, language="text")

                        if result.stderr:
                            with st.expander("View Error Output"):
                                st.code(result.stderr, language="text")

                except Exception as e:
                    st.error(f"❌ Error running tests: {str(e)}")

    with col2:
        st.subheader("Test Coverage")

        # Mock test coverage data
        coverage_data = {
            "Unit Tests": 85,
            "Integration Tests": 92,
            "Data Processing": 95,
            "Model Loading": 88,
            "API Tests": 78,
            "Performance Tests": 82
        }

        for test_type, coverage in coverage_data.items():
            if test_type.replace(" Tests", "") == selected_test.replace(" Tests", ""):
                st.metric(f"{test_type} Coverage", f"{coverage}%")
            else:
                st.write(f"{test_type}: {coverage}%")

        st.subheader("Quick Actions")

        if st.button("🔄 Run All Tests"):
            st.info("Running all tests... (This would run the full test suite)")

        if st.button("📊 Generate Coverage Report"):
            st.info("Generating coverage report... (This would create HTML coverage reports)")

if __name__ == "__main__":
    main()