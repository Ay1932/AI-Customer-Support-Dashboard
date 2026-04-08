"""
Unit tests for the AI Customer Support Dashboard
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import preprocess_text

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class TestPreprocessing:
    """Test text preprocessing"""

    def test_basic(self):
        result = preprocess_text("Hello World! This is a TEST 123.")
        assert isinstance(result, str)
        assert result == result.lower()
        assert "!" not in result
        assert "123" not in result

    def test_empty(self):
        assert preprocess_text("") == ""

    def test_nan(self):
        assert preprocess_text(np.nan) == ""

    def test_stopwords_removed(self):
        result = preprocess_text("this is a very important test")
        assert "important" in result
        assert "test" in result


class TestModelFiles:
    """Test required model files exist"""

    def test_vectorizer(self):
        assert os.path.exists(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

    def test_category_model(self):
        assert os.path.exists(os.path.join(MODEL_DIR, 'category_model.pkl'))

    def test_priority_model(self):
        assert os.path.exists(os.path.join(MODEL_DIR, 'priority_model.pkl'))

    def test_encoders(self):
        assert os.path.exists(os.path.join(MODEL_DIR, 'category_encoder.pkl'))
        assert os.path.exists(os.path.join(MODEL_DIR, 'priority_encoder.pkl'))

    def test_metrics(self):
        path = os.path.join(MODEL_DIR, 'model_metrics.json')
        assert os.path.exists(path)
        with open(path, 'r') as f:
            metrics = json.load(f)
        assert 'category' in metrics
        assert 'priority' in metrics


class TestPrediction:
    """Test model prediction pipeline"""

    def test_end_to_end(self):
        import joblib
        from pathlib import Path

        try:
            vectorizer = joblib.load(Path(MODEL_DIR) / 'tfidf_vectorizer.pkl')
            cat_model = joblib.load(Path(MODEL_DIR) / 'category_model.pkl')
            pri_model = joblib.load(Path(MODEL_DIR) / 'priority_model.pkl')
            cat_encoder = joblib.load(Path(MODEL_DIR) / 'category_encoder.pkl')
            pri_encoder = joblib.load(Path(MODEL_DIR) / 'priority_encoder.pkl')
        except FileNotFoundError:
            pytest.skip("Model files not found — run training first")

        text = preprocess_text("My login is not working")
        vec = vectorizer.transform([text])

        cat_label = cat_encoder.inverse_transform([cat_model.predict(vec)[0]])[0]
        pri_label = pri_encoder.inverse_transform([pri_model.predict(vec)[0]])[0]

        assert isinstance(cat_label, str) and len(cat_label) > 0
        assert isinstance(pri_label, str) and len(pri_label) > 0


class TestData:
    """Test sample data file"""

    def test_csv_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, 'sample_tickets.csv'))

    def test_csv_columns(self):
        df = pd.read_csv(os.path.join(DATA_DIR, 'sample_tickets.csv'))
        for col in ['Ticket ID', 'Ticket Type', 'Subject', 'Description', 'Priority', 'Status']:
            assert col in df.columns

    def test_csv_not_empty(self):
        df = pd.read_csv(os.path.join(DATA_DIR, 'sample_tickets.csv'))
        assert len(df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])