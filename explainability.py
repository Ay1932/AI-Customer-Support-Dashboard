"""
Model Explainability and Monitoring
Uses SHAP and LIME to explain model predictions
Includes model performance monitoring and drift detection
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import explainability libraries (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME not available. Install with: pip install lime")

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess_data, preprocess_text

class ModelExplainer:
    """Class for explaining model predictions"""

    def __init__(self, model_path, vectorizer_path, encoder_path, model_type="category"):
        """Initialize explainer with model paths"""
        self.model_type = model_type

        # Load model components
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.encoder = joblib.load(encoder_path)
            print(f"✓ Loaded {model_type} model components")
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            self.model = None
            self.vectorizer = None
            self.encoder = None

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None

        if SHAP_AVAILABLE and self.model:
            try:
                # Create SHAP explainer (using sampling for large datasets)
                self.shap_explainer = shap.Explainer(self.model)
                print(f"✓ Initialized SHAP explainer for {model_type}")
            except Exception as e:
                print(f"⚠️  SHAP explainer failed: {e}")

        if LIME_AVAILABLE:
            try:
                self.lime_explainer = LimeTextExplainer(class_names=self.encoder.classes_)
                print(f"✓ Initialized LIME explainer for {model_type}")
            except Exception as e:
                print(f"⚠️  LIME explainer failed: {e}")

    def explain_prediction_shap(self, text, max_evals=100):
        """Explain prediction using SHAP"""
        if not SHAP_AVAILABLE or not self.shap_explainer:
            return None

        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            text_vector = self.vectorizer.transform([processed_text])

            # Convert to dense array for SHAP
            text_dense = text_vector.toarray()

            # Get SHAP values
            shap_values = self.shap_explainer(text_dense, max_evals=max_evals)

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Create explanation summary
            explanation = {
                'prediction': self.encoder.inverse_transform([self.model.predict(text_vector)[0]])[0],
                'prediction_proba': self.model.predict_proba(text_vector)[0].tolist(),
                'top_features': []
            }

            # Get top contributing features
            if hasattr(shap_values, 'values'):
                mean_shap = np.mean(np.abs(shap_values.values), axis=0)
                top_indices = np.argsort(mean_shap)[-10:][::-1]  # Top 10 features

                for idx in top_indices:
                    if idx < len(feature_names):
                        explanation['top_features'].append({
                            'feature': feature_names[idx],
                            'importance': float(mean_shap[idx])
                        })

            return explanation

        except Exception as e:
            print(f"❌ SHAP explanation failed: {e}")
            return None

    def explain_prediction_lime(self, text, num_features=10):
        """Explain prediction using LIME"""
        if not LIME_AVAILABLE or not self.lime_explainer:
            return None

        try:
            def predict_proba_func(texts):
                """Prediction function for LIME"""
                processed_texts = [preprocess_text(t) for t in texts]
                text_vectors = self.vectorizer.transform(processed_texts)
                return self.model.predict_proba(text_vectors)

            # Get LIME explanation
            exp = self.lime_explainer.explain_instance(
                text,
                predict_proba_func,
                num_features=num_features
            )

            # Format explanation
            explanation = {
                'prediction': exp.predicted_value,
                'prediction_class': self.encoder.classes_[exp.predicted_value],
                'features': []
            }

            # Get feature contributions
            for feature, weight in exp.as_list():
                explanation['features'].append({
                    'feature': feature,
                    'weight': weight
                })

            return explanation

        except Exception as e:
            print(f"❌ LIME explanation failed: {e}")
            return None

    def explain_prediction(self, text, method='both'):
        """Explain prediction using available methods"""
        explanations = {}

        if method in ['shap', 'both'] and SHAP_AVAILABLE:
            explanations['shap'] = self.explain_prediction_shap(text)

        if method in ['lime', 'both'] and LIME_AVAILABLE:
            explanations['lime'] = self.explain_prediction_lime(text)

        return explanations

class ModelMonitor:
    """Class for monitoring model performance over time"""

    def __init__(self, model_dir='models'):
        """Initialize model monitor"""
        self.model_dir = model_dir
        self.monitoring_history = []
        self.load_history()

    def load_history(self):
        """Load monitoring history from file"""
        history_file = os.path.join(self.model_dir, 'monitoring_history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.monitoring_history = json.load(f)
                print("✓ Loaded monitoring history")
            except Exception as e:
                print(f"⚠️  Could not load monitoring history: {e}")
                self.monitoring_history = []

    def save_history(self):
        """Save monitoring history to file"""
        os.makedirs(self.model_dir, exist_ok=True)
        history_file = os.path.join(self.model_dir, 'monitoring_history.json')

        with open(history_file, 'w') as f:
            json.dump(self.monitoring_history, f, indent=2, default=str)

    def evaluate_model_performance(self, X_test, y_test, model_name="current"):
        """Evaluate current model performance"""
        try:
            # Load current models
            category_model = joblib.load(os.path.join(self.model_dir, 'category_model.pkl'))
            priority_model = joblib.load(os.path.join(self.model_dir, 'priority_model.pkl'))
            vectorizer = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))

            # Transform test data
            X_test_tfidf = vectorizer.transform(X_test)

            # Evaluate category model
            cat_pred = category_model.predict(X_test_tfidf)
            cat_accuracy = accuracy_score(y_test[0], cat_pred)  # y_test[0] for category

            # Evaluate priority model
            pri_pred = priority_model.predict(X_test_tfidf)
            pri_accuracy = accuracy_score(y_test[1], pri_pred)  # y_test[1] for priority

            # Record evaluation
            evaluation = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'category_accuracy': cat_accuracy,
                'priority_accuracy': pri_accuracy,
                'sample_size': len(X_test),
                'category_report': classification_report(y_test[0], cat_pred, output_dict=True),
                'priority_report': classification_report(y_test[1], pri_pred, output_dict=True)
            }

            self.monitoring_history.append(evaluation)
            self.save_history()

            return evaluation

        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            return None

    def detect_drift(self, window_days=30):
        """Detect performance drift over time"""
        if len(self.monitoring_history) < 2:
            return None

        # Get recent evaluations
        recent_evals = [e for e in self.monitoring_history
                       if (datetime.now() - datetime.fromisoformat(e['timestamp'])).days <= window_days]

        if len(recent_evals) < 2:
            return None

        # Calculate drift metrics
        baseline = recent_evals[0]  # First evaluation in window
        latest = recent_evals[-1]   # Most recent evaluation

        drift = {
            'window_days': window_days,
            'baseline_timestamp': baseline['timestamp'],
            'latest_timestamp': latest['timestamp'],
            'category_accuracy_drift': latest['category_accuracy'] - baseline['category_accuracy'],
            'priority_accuracy_drift': latest['priority_accuracy'] - baseline['priority_accuracy'],
            'drift_detected': abs(latest['category_accuracy'] - baseline['category_accuracy']) > 0.05 or
                            abs(latest['priority_accuracy'] - baseline['priority_accuracy']) > 0.05
        }

        return drift

    def get_performance_trends(self):
        """Get performance trends over time"""
        if not self.monitoring_history:
            return None

        # Sort by timestamp
        sorted_history = sorted(self.monitoring_history, key=lambda x: x['timestamp'])

        trends = {
            'timestamps': [e['timestamp'] for e in sorted_history],
            'category_accuracy': [e['category_accuracy'] for e in sorted_history],
            'priority_accuracy': [e['priority_accuracy'] for e in sorted_history]
        }

        return trends

def create_explainability_demo():
    """Create a demonstration of model explainability"""

    print("🔍 Creating Model Explainability Demo")
    print("=" * 50)

    # Initialize explainers
    category_explainer = ModelExplainer(
        'models/category_model.pkl',
        'models/tfidf_vectorizer.pkl',
        'models/category_encoder.pkl',
        'category'
    )

    priority_explainer = ModelExplainer(
        'models/priority_model.pkl',
        'models/tfidf_vectorizer.pkl',
        'models/priority_encoder.pkl',
        'priority'
    )

    # Sample texts for explanation
    sample_texts = [
        "My account login is not working, I can't access my dashboard",
        "I received the wrong item in my order, it's completely different from what I ordered",
        "The website is running very slowly, pages take forever to load",
        "I need to update my billing information for recurring payments"
    ]

    explanations = []

    for text in sample_texts:
        print(f"\n📝 Explaining: '{text[:50]}...'")

        # Get explanations
        cat_explanation = category_explainer.explain_prediction(text)
        pri_explanation = priority_explainer.explain_prediction(text)

        explanation = {
            'text': text,
            'category_explanation': cat_explanation,
            'priority_explanation': pri_explanation
        }

        explanations.append(explanation)

        # Print summary
        if cat_explanation.get('shap'):
            pred = cat_explanation['shap'].get('prediction', 'Unknown')
            print(f"   Category: {pred}")

        if pri_explanation.get('shap'):
            pred = pri_explanation['shap'].get('prediction', 'Unknown')
            print(f"   Priority: {pred}")

    # Save explanations
    os.makedirs('models', exist_ok=True)
    with open('models/explanations_demo.json', 'w') as f:
        json.dump(explanations, f, indent=2, default=str)

    print("✓ Saved explanations to models/explanations_demo.json")

    return explanations

def run_model_monitoring():
    """Run model monitoring and evaluation"""

    print("📊 Running Model Monitoring")
    print("=" * 40)

    # Initialize monitor
    monitor = ModelMonitor()

    # Load test data for evaluation
    try:
        X, y_category, y_priority = load_and_preprocess_data()

        # Create a small test set
        _, X_test, _, y_cat_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
        _, _, _, y_pri_test = train_test_split(X, y_priority, test_size=0.2, random_state=42)

        # Evaluate current performance
        evaluation = monitor.evaluate_model_performance(
            X_test, [y_cat_test, y_pri_test], "current_evaluation"
        )

        if evaluation:
            print(".3f"            print(".3f"
        # Check for drift
        drift = monitor.detect_drift()
        if drift:
            print("
📈 Drift Detection:"            print(".3f"            print(".3f"            if drift['drift_detected']:
                print("   ⚠️  Performance drift detected!")
            else:
                print("   ✓ No significant drift detected")

        # Get trends
        trends = monitor.get_performance_trends()
        if trends:
            print("
📈 Performance Trends:"            print(f"   Evaluations: {len(trends['timestamps'])}")
            if len(trends['category_accuracy']) > 1:
                cat_trend = trends['category_accuracy'][-1] - trends['category_accuracy'][0]
                pri_trend = trends['priority_accuracy'][-1] - trends['priority_accuracy'][0]
                print(".3f"                print(".3f"
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")

if __name__ == "__main__":
    print("🤖 AI Customer Support - Model Explainability & Monitoring")
    print("=" * 60)

    # Run explainability demo
    create_explainability_demo()

    # Run monitoring
    run_model_monitoring()

    print("\n🎉 Explainability and monitoring demo completed!")
    print("📁 Check models/ directory for results")