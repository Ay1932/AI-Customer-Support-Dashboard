import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import json
import requests

# Import shared preprocessing (single source of truth)
from data_preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
METRICS_FILE = MODEL_DIR / 'model_metrics.json'


# ── Custom CSS Theme ─────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border: 1px solid #667eea44;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #8b95b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }

    /* Headers */
    h1 { font-weight: 700 !important; }
    h2 { font-weight: 600 !important; color: #e0e4f0 !important; }
    h3 { font-weight: 600 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }

    hr { border-color: #667eea33 !important; }
</style>
"""


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    vectorizer = joblib.load(MODEL_DIR / 'tfidf_vectorizer.pkl')
    category_model = joblib.load(MODEL_DIR / 'category_model.pkl')
    priority_model = joblib.load(MODEL_DIR / 'priority_model.pkl')
    category_encoder = joblib.load(MODEL_DIR / 'category_encoder.pkl')
    priority_encoder = joblib.load(MODEL_DIR / 'priority_encoder.pkl')
    return vectorizer, category_model, priority_model, category_encoder, priority_encoder


def predict_ticket(text, vectorizer, category_model, priority_model, cat_encoder, pri_encoder):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    category_pred = category_model.predict(vectorized_text)[0]
    priority_pred = priority_model.predict(vectorized_text)[0]
    category_label = cat_encoder.inverse_transform([category_pred])[0]
    priority_label = pri_encoder.inverse_transform([priority_pred])[0]
    return category_label, priority_label


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


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Customer Support Dashboard",
        page_icon="🎫",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🎫 AI Customer Support Dashboard")

    # Load models and data
    try:
        vectorizer, category_model, priority_model, cat_encoder, pri_encoder = load_models()
        df = load_ticket_data()
        metrics = load_model_metrics()
    except FileNotFoundError as exc:
        st.error(f"Models or data not found: {exc.filename}. Please run preprocessing and training first.")
        return

    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    pages = ["Analytics Dashboard", "Data Explorer", "Ticket Prediction", "API Testing"]
    page = st.sidebar.selectbox("Choose a page", pages)

    if page == "Analytics Dashboard":
        show_analytics_dashboard(df, metrics)
    elif page == "Data Explorer":
        show_data_explorer(df)
    elif page == "Ticket Prediction":
        show_ticket_prediction(vectorizer, category_model, priority_model, cat_encoder, pri_encoder)
    elif page == "API Testing":
        show_api_testing()


# ── Page: Analytics Dashboard ─────────────────────────────────────────────────

def show_analytics_dashboard(df, metrics):
    st.header("📊 Analytics Dashboard")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    total_tickets = len(df)
    open_tickets = len(df[df['Status'] == 'Open'])
    avg_resolution_time = df['Resolved Date'].sub(df['Created Date']).dt.days.mean()
    high_priority_open = len(df[(df['Status'] == 'Open') & (df['Priority'] == 'High')])

    col1.metric("Total Tickets", total_tickets)
    col2.metric("Open Tickets", open_tickets)
    col3.metric("🚨 High Priority Open", high_priority_open)
    col4.metric("Avg Resolution (days)", f"{avg_resolution_time:.1f}")

    st.markdown("---")

    # Charts Row
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.pie(df, names='Ticket Type', title='Ticket Types Distribution',
                     color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df['Priority'].value_counts(), title='Priority Distribution',
                     labels={'value': 'Count', 'index': 'Priority'},
                     color_discrete_sequence=['#667eea'])
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.bar(df['Status'].value_counts(), title='Ticket Status',
                     labels={'value': 'Count', 'index': 'Status'},
                     color_discrete_sequence=['#764ba2'])
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Time series
    st.subheader("📈 Ticket Volume Over Time")
    daily_tickets = df.groupby(df['Created Date'].dt.date).size().reset_index(name='count')
    fig = px.line(daily_tickets, x='Created Date', y='count', title='Daily Ticket Volume',
                  markers=True, color_discrete_sequence=['#667eea'])
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # Category Analysis
    st.subheader("📂 Category Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df['Ticket Type'].value_counts(), title='Tickets by Category',
                     labels={'value': 'Count', 'index': 'Category'},
                     color_discrete_sequence=['#667eea'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_resolved = df[df['Status'] == 'Resolved'].copy()
        df_resolved['resolution_days'] = (df_resolved['Resolved Date'] - df_resolved['Created Date']).dt.days
        avg_by_category = df_resolved.groupby('Ticket Type')['resolution_days'].mean().sort_values(ascending=False)
        fig = px.bar(avg_by_category, title='Avg Resolution Time by Category (days)',
                     labels={'value': 'Days', 'index': 'Category'},
                     color_discrete_sequence=['#764ba2'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Priority vs Status Heatmap
    st.subheader("🔥 Priority vs Status Matrix")
    priority_status = pd.crosstab(df['Priority'], df['Status'])
    fig = px.imshow(priority_status, labels=dict(x='Status', y='Priority', color='Count'),
                    title='Tickets by Priority and Status', color_continuous_scale='Purples')
    st.plotly_chart(fig, use_container_width=True)

    # Export data
    st.subheader("📥 Export Data")
    csv = df.to_csv(index=False)
    st.download_button(label="Download all tickets as CSV", data=csv,
                       file_name="tickets_export.csv", mime="text/csv")

    # Model performance
    if metrics:
        show_model_performance(metrics)


def show_model_performance(metrics):
    st.subheader("🧠 Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Category Model**")
        accuracy = metrics.get('category', {}).get('accuracy')
        if accuracy is not None:
            st.metric("Accuracy", f"{accuracy:.2f}")

    with col2:
        st.markdown("**Priority Model**")
        accuracy = metrics.get('priority', {}).get('accuracy')
        if accuracy is not None:
            st.metric("Accuracy", f"{accuracy:.2f}")


# ── Page: Data Explorer ───────────────────────────────────────────────────────

def show_data_explorer(df):
    st.header("📁 Data Explorer")
    st.write("Filter, search, and inspect ticket records.")

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

    display_cols = ['Ticket ID', 'Ticket Type', 'Subject', 'Priority', 'Status', 'Created Date', 'Resolved Date']
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)

    with st.expander("Show ticket descriptions"):
        st.dataframe(filtered[['Ticket ID', 'Subject', 'Description']], use_container_width=True, height=400)

    if len(filtered) > 0:
        csv = filtered.to_csv(index=False)
        st.download_button(label=f"Download {len(filtered)} filtered tickets as CSV",
                           data=csv, file_name="filtered_tickets_export.csv", mime="text/csv")


# ── Page: Ticket Prediction ──────────────────────────────────────────────────

def show_ticket_prediction(vectorizer, category_model, priority_model, cat_encoder, pri_encoder):
    st.header("🔮 Ticket Prediction")
    st.write("Enter ticket details to predict category and priority:")

    subject = st.text_input("Ticket Subject")
    description = st.text_area("Ticket Description")

    if st.button("Predict", type="primary"):
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


# ── Page: API Testing ─────────────────────────────────────────────────────────

def show_api_testing():
    st.header("🌐 API Testing")
    st.write("Test the REST API for programmatic access to predictions.")

    api_url = st.text_input("API Base URL", "http://localhost:8000")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Check API Health"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        st.success("✅ API is healthy")
                    else:
                        st.warning("⚠️ API running but models may not be loaded")
                    st.json(data)
                else:
                    st.error(f"❌ Health check failed: {response.status_code}")
            except requests.exceptions.RequestException:
                st.error("❌ Cannot connect to API")
                st.info("Start the API with: `uvicorn api:app --reload`")

    with col2:
        if st.button("Open API Docs"):
            st.markdown(f"[API Documentation]({api_url}/docs)")

    st.markdown("---")
    st.subheader("Test Prediction")

    subject = st.text_input("Ticket Subject", key="api_subject")
    description = st.text_area("Ticket Description", key="api_desc")

    if st.button("Send Prediction Request", type="primary"):
        if subject and description:
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json={"subject": subject, "description": description},
                    timeout=10,
                )
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
                else:
                    st.error(f"❌ Request failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Cannot connect: {e}")
        else:
            st.warning("Please enter both subject and description.")

    # Quick code example
    with st.expander("Python Usage Example"):
        st.code("""
import requests

response = requests.post("http://localhost:8000/predict", json={
    "subject": "Login not working",
    "description": "I cannot access my account after password reset"
})

result = response.json()
print(f"Category: {result['category']}, Priority: {result['priority']}")
        """, language="python")


if __name__ == "__main__":
    main()