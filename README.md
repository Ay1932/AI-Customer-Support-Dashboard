# AI-Based Customer Support Ticket Prioritization and Analytics Dashboard

This project implements an AI-powered system for customer support ticket classification, prioritization, and analytics visualization. It uses machine learning to automatically categorize tickets and predict their priority levels, helping support teams respond more efficiently.

## Features

- **Automatic Ticket Classification**: Classifies tickets into categories (Technical Issue, Billing, Account Problem, Delivery Issue)
- **Priority Prediction**: Predicts ticket priority (Low, Medium, High) based on content
- **Analytics Dashboard**: Visualizes ticket volume, trends, status, and priority distribution
- **Data Explorer**: Filter and inspect tickets by type and status
- **Model Performance Summary**: Displays evaluation metrics for category and priority models
- **Interactive Prediction Interface**: Allows users to input new tickets and get instant predictions

## Technology Stack

- **Python** - Core programming language
- **Scikit-learn** - Machine learning models
- **Streamlit** - Web dashboard and interface
- **Pandas/Numpy** - Data processing
- **NLTK** - Natural language processing
- **Plotly** - Data visualization
- **TF-IDF** - Text feature extraction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-support-dashboard
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Setup Environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Process Data & Train Models**:
```bash
python data_preprocessing.py
python train_models.py
```

3. **Run Dashboard**:
```bash
streamlit run app.py
# Or on Windows: double-click run_dashboard.bat
```

The dashboard will be available at `http://localhost:8501`

## Project Structure

```
customer_support_dashboard/
├── data/
│   └── sample_tickets.csv          # Sample ticket dataset
├── models/                         # Trained models, preprocessors, and metrics
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── category_encoder.pkl
│   ├── priority_encoder.pkl
│   └── model_metrics.json
├── data_preprocessing.py           # Data cleaning and preprocessing
├── train_models.py                 # Model training script
├── app.py                          # Streamlit dashboard application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Data Processing Pipeline

1. **Text Preprocessing**:
   - Lowercase conversion
   - Special character removal
   - Stopword removal
   - Lemmatization

2. **Feature Extraction**:
   - TF-IDF vectorization with n-grams
   - Maximum 1000 features

3. **Model Training**:
   - Category Classification: Random Forest
   - Priority Prediction: Logistic Regression
   - Evaluation metrics saved for dashboard display

## Analytics Features

- Ticket volume trends over time
- Category and priority distributions
- Resolution time analysis
- Status tracking (Open/Resolved)

## Prediction Interface

Users can input ticket subject and description to get:
- Predicted category
- Predicted priority level
- Color-coded priority indicators

## Model Performance

The models are evaluated using:
- Accuracy score
- Precision, Recall, F1-score
- Classification reports

## Future Enhancements

- Integration with real ticketing systems (Jira, Zendesk)
- Real-time model retraining using an updated ticket stream
- Advanced NLP techniques (BERT, transformers, or fastText)
- Multi-language support for non-English tickets
- Automated response suggestion and ticket routing
- Add a business impact dashboard for SLA breaches and backlog risk
- Add explainability (SHAP/LIME) for why a ticket was prioritized high

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Sample customer support tickets
- Libraries: Scikit-learn, NLTK, Streamlit, Plotly