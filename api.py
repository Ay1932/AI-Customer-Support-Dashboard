"""
FastAPI REST API for AI Customer Support Dashboard
Provides endpoints for ticket classification and priority prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
from typing import Optional
import os
from data_preprocessing import preprocess_text

# Initialize FastAPI app
app = FastAPI(
    title="AI Customer Support API",
    description="REST API for ticket classification and priority prediction",
    version="1.0.0"
)

# Load models and preprocessors
try:
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    category_model = joblib.load(os.path.join(model_dir, 'category_model.pkl'))
    priority_model = joblib.load(os.path.join(model_dir, 'priority_model.pkl'))
    tfidf_vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    category_encoder = joblib.load(os.path.join(model_dir, 'category_encoder.pkl'))
    priority_encoder = joblib.load(os.path.join(model_dir, 'priority_encoder.pkl'))
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    category_model = None
    priority_model = None
    tfidf_vectorizer = None
    category_encoder = None
    priority_encoder = None

# Pydantic models for request/response
class TicketRequest(BaseModel):
    subject: str
    description: str

class PredictionResponse(BaseModel):
    category: str
    priority: str
    confidence_category: float
    confidence_priority: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Customer Support API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if all([category_model, priority_model, tfidf_vectorizer]) else "unhealthy",
        models_loaded=all([category_model, priority_model, tfidf_vectorizer]),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_ticket(ticket: TicketRequest):
    """
    Predict ticket category and priority

    Takes a ticket with subject and description, returns predicted category and priority
    with confidence scores.
    """
    if not all([category_model, priority_model, tfidf_vectorizer]):
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Combine subject and description
        full_text = f"{ticket.subject} {ticket.description}"

        # Preprocess text
        processed_text = preprocess_text(full_text)

        # Transform text to features
        text_features = tfidf_vectorizer.transform([processed_text])

        # Predict category
        category_pred = category_model.predict(text_features)[0]
        category_prob = np.max(category_model.predict_proba(text_features)[0])
        category_label = category_encoder.inverse_transform([category_pred])[0]

        # Predict priority
        priority_pred = priority_model.predict(text_features)[0]
        priority_prob = np.max(priority_model.predict_proba(text_features)[0])
        priority_label = priority_encoder.inverse_transform([priority_pred])[0]

        return PredictionResponse(
            category=category_label,
            priority=priority_label,
            confidence_category=round(float(category_prob), 3),
            confidence_priority=round(float(priority_prob), 3)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/categories", tags=["Metadata"])
async def get_categories():
    """Get available ticket categories"""
    if not category_encoder:
        raise HTTPException(status_code=503, detail="Category encoder not loaded")

    return {
        "categories": category_encoder.classes_.tolist()
    }

@app.get("/priorities", tags=["Metadata"])
async def get_priorities():
    """Get available priority levels"""
    if not priority_encoder:
        raise HTTPException(status_code=503, detail="Priority encoder not loaded")

    return {
        "priorities": priority_encoder.classes_.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)