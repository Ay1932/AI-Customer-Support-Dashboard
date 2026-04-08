import joblib
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

def train_category_classifier(X_train, y_train):
    """Train category classification model"""
    # Use Random Forest for better performance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_priority_classifier(X_train, y_train):
    """Train priority prediction model"""
    # Use Logistic Regression for priority (ordinal)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load preprocessed data
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = joblib.load(DATA_DIR / 'category_data.pkl')
    X_train_pri, X_test_pri, y_train_pri, y_test_pri = joblib.load(DATA_DIR / 'priority_data.pkl')

    # Train category classifier
    category_model = train_category_classifier(X_train_cat, y_train_cat)
    evaluate_model(category_model, X_test_cat, y_test_cat, "Category Classifier")

    # Train priority classifier
    priority_model = train_priority_classifier(X_train_pri, y_train_pri)
    evaluate_model(priority_model, X_test_pri, y_test_pri, "Priority Classifier")

    # Save models
    joblib.dump(category_model, MODEL_DIR / 'category_model.pkl')
    joblib.dump(priority_model, MODEL_DIR / 'priority_model.pkl')

    # Save evaluation metrics for dashboard
    metrics = {
        'category': classification_report(y_test_cat, category_model.predict(X_test_cat), output_dict=True),
        'priority': classification_report(y_test_pri, priority_model.predict(X_test_pri), output_dict=True)
    }
    metrics['category']['accuracy'] = accuracy_score(y_test_cat, category_model.predict(X_test_cat))
    metrics['priority']['accuracy'] = accuracy_score(y_test_pri, priority_model.predict(X_test_pri))

    with open(MODEL_DIR / 'model_metrics.json', 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("\nModels trained and saved successfully!")