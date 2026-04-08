# AI-Based Customer Support Ticket Prioritization and Analytics Dashboard

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Ay1932/AI-Customer-Support-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Ay1932/AI-Customer-Support-Dashboard/actions/workflows/ci.yml)

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

### Option 1: Local Development (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/Ay1932/AI-Customer-Support-Dashboard.git
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

### Option 2: Using Docker

If you have Docker installed, you can run the application in a container:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t customer-support-dashboard .
docker run -p 8501:8501 customer-support-dashboard
```

### Option 3: Using Make (Linux/Mac)

If you have `make` installed, you can use the provided Makefile for common tasks:

```bash
make setup    # Create virtual environment
make install  # Install dependencies
make train    # Train models
make run      # Run dashboard
make all      # Do everything
```

## Quick Start

### For Local Development

1. **Setup Environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
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
# Or use Make: make run
```

4. **View Feature Demo** (Optional):
```bash
streamlit run demo.py
```

### For Advanced Features

**Train Advanced Models with Hyperparameter Tuning**:
```bash
make train-advanced
# or
python advanced_training.py
```

**Run Model Explainability Analysis**:
```bash
make explain
# or
python explainability.py
```

**Run REST API**:
```bash
make run-api
# or
uvicorn api:app --reload
```

### For Docker Users

```bash
docker-compose up --build
```

The dashboard will be available at `http://localhost:8501`

## Project Structure

```
customer_support_dashboard/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI pipeline
├── data/
│   └── sample_tickets.csv           # Sample ticket dataset
├── models/                          # Trained models, preprocessors, and metrics
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── category_encoder.pkl
│   ├── priority_encoder.pkl
│   ├── model_metrics.json
│   ├── advanced_*.pkl              # Advanced models (optional)
│   ├── explanations_demo.json      # Model explanations
│   └── monitoring_history.json     # Performance monitoring
├── tests/                           # Unit and integration tests
│   ├── __init__.py
│   └── test_main.py
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── CODE_OF_CONDUCT.md              # Community code of conduct
├── CONTRIBUTING.md                 # Contribution guidelines
├── Dockerfile                       # Docker container configuration
├── LICENSE                          # MIT license
├── Makefile                         # Development automation tasks
├── README.md                        # Project documentation
├── advanced_training.py             # Advanced model training with tuning
├── api.py                           # FastAPI REST API
├── app.py                           # Streamlit dashboard application
├── data_preprocessing.py            # Data cleaning and preprocessing
├── docker-compose.yml              # Docker Compose configuration
├── explainability.py                # Model explainability and monitoring
├── pyproject.toml                   # Modern Python packaging
├── requirements.txt                 # Python dependencies
├── run_dashboard.bat               # Windows batch script to run dashboard
└── setup.py                         # Traditional Python packaging
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

## Advanced Features

### 🤖 **Advanced Model Training**
- **Hyperparameter Tuning**: Grid search optimization for all models
- **Model Comparison**: Compare Random Forest, SVM, Gradient Boosting, Logistic Regression, and Naive Bayes
- **Ensemble Methods**: Voting classifiers combining multiple models
- **Cross-Validation**: Robust evaluation with k-fold cross-validation

```bash
python advanced_training.py
```

### 🔍 **Model Explainability**
- **SHAP Integration**: Understand feature importance and prediction explanations
- **LIME Integration**: Local interpretable model-agnostic explanations
- **Prediction Confidence**: Confidence scores for all predictions
- **Feature Analysis**: Identify which words/phrases influence predictions

```bash
python explainability.py
```

### 🌐 **REST API**
- **FastAPI Backend**: Production-ready REST API
- **Interactive Documentation**: Auto-generated API docs at `/docs`
- **Prediction Endpoints**: Programmatic access to model predictions
- **Health Monitoring**: API health checks and model status

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# API docs available at: http://localhost:8000/docs
```

### 📊 **Model Monitoring**
- **Performance Tracking**: Monitor model accuracy over time
- **Drift Detection**: Identify when model performance degrades
- **Automated Evaluation**: Regular model assessment on new data
- **Historical Analysis**: Track performance trends and metrics

### 🧪 **Comprehensive Testing**
- **Unit Tests**: Test all core functions and classes
- **Integration Tests**: Test model loading and prediction pipelines
- **Code Coverage**: Track test coverage with pytest-cov
- **CI/CD Integration**: Automated testing on every commit

```bash
pytest --cov=. --cov-report=html
```

### 🐳 **Containerization**
- **Docker Support**: Run anywhere with Docker
- **Docker Compose**: Easy multi-container deployment
- **Production Ready**: Optimized for deployment

```bash
docker-compose up --build
```

## Contributing

Please read our [Contributing Guide](CONTRIBUTING.md) for detailed information on how to contribute to this project.

We welcome contributions of all kinds, including:
- Bug reports and fixes
- Feature requests and implementations
- Documentation improvements
- Code optimizations

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all contributors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Sample customer support tickets
- Libraries: Scikit-learn, NLTK, Streamlit, Plotly