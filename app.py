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
    page = st.sidebar.selectbox("Choose a page", ["Analytics Dashboard", "Data Explorer", "Ticket Prediction"])

    if page == "Analytics Dashboard":
        show_analytics_dashboard(df, metrics)
    elif page == "Data Explorer":
        show_data_explorer(df)
    else:
        show_ticket_prediction(vectorizer, category_model, priority_model, cat_encoder, pri_encoder)

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

if __name__ == "__main__":
    main()