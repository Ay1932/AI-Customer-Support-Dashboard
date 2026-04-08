# 🎫 AI Customer Support Dashboard

An AI-powered dashboard for customer support ticket classification and analytics. Built with **Streamlit**, **scikit-learn**, and **FastAPI**.
Streamlit: https://ai-customer-support-dashboard.streamlit.app/
The system uses machine learning (Random Forest + Logistic Regression) to automatically predict ticket **category** and **priority** from the ticket text.

---

## Features

### 📊 Analytics Dashboard
- Key metrics: total tickets, open tickets, high-priority open, avg resolution time
- Interactive charts: ticket types, priority distribution, status breakdown
- Time series: daily ticket volume trends
- Heatmap: priority vs status matrix
- CSV export

### 📁 Data Explorer
- Filter by type, status, priority
- Full-text search in ticket subjects
- Sortable data table with export

### 🔮 Ticket Prediction
- Enter a ticket subject + description
- AI predicts the **category** (Technical Issue, Billing, Account Problem, Delivery Issue)
- AI predicts the **priority** (High, Medium, Low)

### 🌐 REST API
- FastAPI-based prediction endpoint
- Health check and metadata endpoints
- Interactive API documentation at `/docs`

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data & train models
```bash
python data_preprocessing.py
python train_models.py
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. (Optional) Run the API
```bash
uvicorn api:app --reload
```

---

## Project Structure

```
customer_support_dashboard/
├── app.py                 # Streamlit dashboard (4 pages)
├── api.py                 # FastAPI REST API
├── data_preprocessing.py  # Text preprocessing & data loading
├── train_models.py        # Model training script
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── run_dashboard.bat      # Windows quick-launch script
├── data/
│   └── sample_tickets.csv # 55 sample support tickets
├── models/
│   ├── category_model.pkl     # Trained category classifier
│   ├── priority_model.pkl     # Trained priority classifier
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   ├── category_encoder.pkl   # Category label encoder
│   ├── priority_encoder.pkl   # Priority label encoder
│   └── model_metrics.json     # Model evaluation metrics
└── tests/
    └── test_main.py       # Unit tests (pytest)
```

---

## API Usage

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "subject": "Login not working",
    "description": "I cannot access my account after password reset"
})

result = response.json()
print(f"Category: {result['category']}")
print(f"Priority: {result['priority']}")
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit |
| Charts | Plotly |
| ML Models | scikit-learn (Random Forest, Logistic Regression) |
| NLP | NLTK (tokenization, stopwords, lemmatization), TF-IDF |
| API | FastAPI + Uvicorn |
| Testing | pytest |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
