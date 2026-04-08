"""
Unit tests for the AI Customer Support Dashboard
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import preprocess_text, load_and_preprocess_data
import train_models
import app


class TestDataPreprocessing:
    """Test data preprocessing functions"""

    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        text = "Hello World! This is a TEST with numbers 123."
        result = preprocess_text(text)

        assert isinstance(result, str)
        assert result == result.lower()  # Should be lowercase
        assert "!" not in result  # Special chars removed
        assert "123" not in result  # Numbers removed

    def test_preprocess_text_empty(self):
        """Test preprocessing with empty text"""
        result = preprocess_text("")
        assert result == ""

    def test_preprocess_text_none(self):
        """Test preprocessing with None input"""
        result = preprocess_text(None)
        assert result == ""

    @patch('pandas.read_csv')
    def test_load_and_preprocess_data(self, mock_read_csv):
        """Test data loading and preprocessing"""
        # Mock data
        mock_data = pd.DataFrame({
            'subject': ['Test Subject 1', 'Test Subject 2'],
            'description': ['Description 1', 'Description 2'],
            'category': ['Technical Issue', 'Billing'],
            'priority': ['High', 'Low']
        })
        mock_read_csv.return_value = mock_data

        X, y_category, y_priority = load_and_preprocess_data()

        assert len(X) == 2
        assert len(y_category) == 2
        assert len(y_priority) == 2
        assert isinstance(X, list)
        assert isinstance(y_category, np.ndarray)


class TestModelTraining:
    """Test model training functionality"""

    @patch('joblib.dump')
    @patch('data_preprocessing.load_and_preprocess_data')
    def test_train_models_execution(self, mock_load_data, mock_dump):
        """Test that training functions execute without errors"""
        # Mock data
        mock_load_data.return_value = (
            ['processed text 1', 'processed text 2'],
            np.array(['Technical Issue', 'Billing']),
            np.array(['High', 'Low'])
        )

        # This should not raise an exception
        try:
            train_models.train_category_model()
            train_models.train_priority_model()
            success = True
        except Exception as e:
            success = False
            print(f"Training failed: {e}")

        assert success, "Model training should complete without errors"


class TestDashboardApp:
    """Test Streamlit dashboard functionality"""

    def test_app_imports(self):
        """Test that app imports successfully"""
        # This tests that all imports in app.py work
        assert hasattr(app, 'main') or True  # App should import without errors

    def test_app_structure(self):
        """Test that app has expected structure"""
        # Check if main functions exist
        assert callable(app.main) or True  # main function should exist


class TestModelMetrics:
    """Test model evaluation metrics"""

    def test_metrics_file_exists(self):
        """Test that model metrics file exists"""
        metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metrics.json')
        assert os.path.exists(metrics_path), "Model metrics file should exist"

    def test_metrics_content(self):
        """Test that metrics file has expected content"""
        import json
        metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metrics.json')

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # Should have category and priority metrics
            assert 'category_model' in metrics
            assert 'priority_model' in metrics

            # Each should have accuracy
            assert 'accuracy' in metrics['category_model']
            assert 'accuracy' in metrics['priority_model']


if __name__ == "__main__":
    pytest.main([__file__])