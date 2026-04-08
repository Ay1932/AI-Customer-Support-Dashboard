"""
Advanced Model Training with Hyperparameter Tuning and Model Comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_and_preprocess_data

def create_advanced_models():
    """Create a dictionary of models to compare"""

    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'NaiveBayes': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
        }
    }

    return models

def perform_grid_search(model, params, X_train, y_train, cv=3):
    """Perform grid search with cross-validation"""
    grid_search = GridSearchCV(
        model,
        params,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def create_ensemble_model(best_models):
    """Create an ensemble model using the best performing individual models"""
    estimators = []
    for name, model in best_models.items():
        estimators.append((name.lower(), model))

    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft'  # Use probability-based voting
    )

    return ensemble

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    # Get prediction probabilities for confidence
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        confidence_scores = np.max(y_prob, axis=1)
    else:
        confidence_scores = np.ones(len(y_test)) * 0.5  # Default confidence

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'average_confidence': np.mean(confidence_scores),
        'predictions': y_pred.tolist(),
        'true_labels': y_test.tolist()
    }

def save_model_comparison_results(results, filename):
    """Save model comparison results to JSON"""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    results_serializable = convert_numpy_types(results)

    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)

def train_advanced_models():
    """Main function to train and compare advanced models"""

    print("🚀 Starting Advanced Model Training and Comparison")
    print("=" * 60)

    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    X, y_category, y_priority = load_and_preprocess_data()

    # Split data
    X_train, X_test, y_cat_train, y_cat_test = train_test_split(
        X, y_category, test_size=0.2, random_state=42, stratify=y_category
    )
    _, _, y_pri_train, y_pri_test = train_test_split(
        X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
    )

    # Create TF-IDF vectorizer
    print("🔧 Creating TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Get available models
    models = create_advanced_models()

    # Results storage
    category_results = {}
    priority_results = {}
    best_category_models = {}
    best_priority_models = {}

    # Train and evaluate category models
    print("\n🏷️  Training Category Classification Models")
    print("-" * 40)

    for model_name, model_config in models.items():
        print(f"\n🔍 Training {model_name}...")

        try:
            # Perform grid search
            best_model, best_params, cv_score = perform_grid_search(
                model_config['model'], model_config['params'],
                X_train_tfidf, y_cat_train
            )

            # Evaluate on test set
            test_results = evaluate_model(best_model, X_test_tfidf, y_cat_test, model_name)

            # Store results
            category_results[model_name] = {
                'best_params': best_params,
                'cv_score': cv_score,
                'test_results': test_results
            }

            best_category_models[model_name] = best_model

            print(".3f"            print(f"   Best params: {best_params}")

        except Exception as e:
            print(f"   ❌ Error training {model_name}: {e}")
            continue

    # Train and evaluate priority models
    print("\n⚡ Training Priority Prediction Models")
    print("-" * 40)

    for model_name, model_config in models.items():
        print(f"\n🔍 Training {model_name}...")

        try:
            # Perform grid search
            best_model, best_params, cv_score = perform_grid_search(
                model_config['model'], model_config['params'],
                X_train_tfidf, y_pri_train
            )

            # Evaluate on test set
            test_results = evaluate_model(best_model, X_test_tfidf, y_pri_test, model_name)

            # Store results
            priority_results[model_name] = {
                'best_params': best_params,
                'cv_score': cv_score,
                'test_results': test_results
            }

            best_priority_models[model_name] = best_model

            print(".3f"            print(f"   Best params: {best_params}")

        except Exception as e:
            print(f"   ❌ Error training {model_name}: {e}")
            continue

    # Create ensemble models
    print("\n🎯 Creating Ensemble Models")
    print("-" * 30)

    if len(best_category_models) >= 3:
        category_ensemble = create_ensemble_model(best_category_models)
        category_ensemble.fit(X_train_tfidf, y_cat_train)
        ensemble_cat_results = evaluate_model(category_ensemble, X_test_tfidf, y_cat_test, "Ensemble")

        category_results['Ensemble'] = {
            'test_results': ensemble_cat_results
        }

        print(".3f"
    if len(best_priority_models) >= 3:
        priority_ensemble = create_ensemble_model(best_priority_models)
        priority_ensemble.fit(X_train_tfidf, y_pri_train)
        ensemble_pri_results = evaluate_model(priority_ensemble, X_test_tfidf, y_pri_test, "Ensemble")

        priority_results['Ensemble'] = {
            'test_results': ensemble_pri_results
        }

        print(".3f"
    # Save results
    print("\n💾 Saving Results...")
    results = {
        'timestamp': datetime.now().isoformat(),
        'category_models': category_results,
        'priority_models': priority_results,
        'data_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train_tfidf.shape[1]
        }
    }

    os.makedirs('models', exist_ok=True)
    save_model_comparison_results(results, 'models/advanced_model_comparison.json')

    # Save best models
    if category_results:
        best_cat_model_name = max(category_results.keys(),
                                 key=lambda x: category_results[x]['test_results']['accuracy'])
        best_category_model = best_category_models.get(best_cat_model_name, best_category_models.get('RandomForest'))

        if best_category_model:
            joblib.dump(best_category_model, 'models/advanced_category_model.pkl')
            print(f"✓ Saved best category model: {best_cat_model_name}")

    if priority_results:
        best_pri_model_name = max(priority_results.keys(),
                                 key=lambda x: priority_results[x]['test_results']['accuracy'])
        best_priority_model = best_priority_models.get(best_pri_model_name, best_priority_models.get('RandomForest'))

        if best_priority_model:
            joblib.dump(best_priority_model, 'models/advanced_priority_model.pkl')
            print(f"✓ Saved best priority model: {best_pri_model_name}")

    # Save advanced vectorizer
    joblib.dump(tfidf, 'models/advanced_tfidf_vectorizer.pkl')
    print("✓ Saved advanced TF-IDF vectorizer")

    print("\n🎉 Advanced model training completed!")
    print("📊 Results saved to models/advanced_model_comparison.json")

    return results

if __name__ == "__main__":
    results = train_advanced_models()

    # Print summary
    print("\n📈 Model Comparison Summary")
    print("=" * 50)

    if 'category_models' in results:
        print("\n🏷️  Category Classification:")
        for model_name, data in results['category_models'].items():
            if 'test_results' in data:
                acc = data['test_results']['accuracy']
                print(".3f"
    if 'priority_models' in results:
        print("\n⚡ Priority Prediction:")
        for model_name, data in results['priority_models'].items():
            if 'test_results' in data:
                acc = data['test_results']['accuracy']
                print(".3f"