.PHONY: help install setup train run clean test lint format

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

setup:  ## Set up development environment
	python -m venv venv
	@echo "Virtual environment created. Run 'venv\Scripts\activate' on Windows or 'source venv/bin/activate' on Linux/Mac"

train:  ## Train the ML models
	python data_preprocessing.py
	python train_models.py

run:  ## Run the Streamlit dashboard
	streamlit run app.py

clean:  ## Clean up generated files
	rm -rf __pycache__
	rm -rf models/*.pkl
	rm -rf data/*.pkl
	rm -rf .pytest_cache

test:  ## Run tests
	python -c "import data_preprocessing; print('✓ Data preprocessing imports successfully')"
	python -c "import train_models; print('✓ Model training imports successfully')"
	python -c "import app; print('✓ Dashboard app imports successfully')"

lint:  ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:  ## Format code with black
	black --line-length 127 .

all: install train run  ## Install dependencies, train models, and run dashboard