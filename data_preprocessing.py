import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def load_data(filepath):
    """Load the ticket dataset"""
    df = pd.read_csv(filepath)
    return df

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def preprocess_data(df):
    """Preprocess the entire dataset"""
    # Combine subject and description
    df['text'] = df['Subject'] + ' ' + df['Description']

    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Encode categorical variables
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['Ticket Type'])

    le_priority = LabelEncoder()
    df['priority_encoded'] = le_priority.fit_transform(df['Priority'])

    # Save encoders
    joblib.dump(le_category, MODEL_DIR / 'category_encoder.pkl')
    joblib.dump(le_priority, MODEL_DIR / 'priority_encoder.pkl')

    return df, le_category, le_priority

def prepare_features(df):
    """Prepare features for modeling"""
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['processed_text'])

    # Save vectorizer
    joblib.dump(vectorizer, MODEL_DIR / 'tfidf_vectorizer.pkl')

    return X

if __name__ == "__main__":
    # Load data
    df = load_data(DATA_DIR / 'sample_tickets.csv')

    # Preprocess data
    df_processed, le_cat, le_pri = preprocess_data(df)

    # Prepare features
    X = prepare_features(df_processed)

    # Split data for category classification
    y_category = df_processed['category_encoded']
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X, y_category, test_size=0.2, random_state=42
    )

    # Split data for priority prediction
    y_priority = df_processed['priority_encoded']
    X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
        X, y_priority, test_size=0.2, random_state=42
    )

    # Save processed data
    joblib.dump((X_train_cat, X_test_cat, y_train_cat, y_test_cat), DATA_DIR / 'category_data.pkl')
    joblib.dump((X_train_pri, X_test_pri, y_train_pri, y_test_pri), DATA_DIR / 'priority_data.pkl')

    print("Data preprocessing completed!")