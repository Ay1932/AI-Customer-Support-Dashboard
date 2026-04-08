.PHONY: help install setup train train-advanced run run-api clean test lint format explain monitor all

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

setup:  ## Set up development environment
	python -m venv venv
	@echo "Virtual environment created. Run 'venv\Scripts\activate' on Windows or 'source venv/bin/activate' on Linux/Mac"

train:  ## Train the basic ML models
	python data_preprocessing.py
	python train_models.py

train-advanced:  ## Train advanced models with hyperparameter tuning
	python data_preprocessing.py
	python advanced_training.py

run:  ## Run the Streamlit dashboard
	streamlit run app.py

run-api:  ## Run the FastAPI REST API
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

explain:  ## Run model explainability analysis
	python explainability.py

monitor:  ## Run model monitoring and evaluation
	python explainability.py

clean:  ## Clean up generated files
	rm -rf __pycache__
	rm -rf models/*.pkl
	rm -rf data/*.pkl
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

test:  ## Run tests
	pytest --cov=. --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:  ## Format code with black
	black --line-length 127 .

docker-build:  ## Build Docker image
	docker build -t customer-support-dashboard .

docker-run:  ## Run Docker container
	docker run -p 8501:8501 customer-support-dashboard

all: install train run  ## Install dependencies, train models, and run dashboard

advanced-all: install train-advanced explain run  ## Full advanced pipeline