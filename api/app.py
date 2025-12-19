# api/app.py
"""
FastAPI service for drug effectiveness prediction.
Loads the trained model and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import shared pipeline components so unpickling works
from drug_pipeline import (
    TextCombiner,
    TextPreprocessor,
    ArrayFlattener,
    build_preprocessing,
    make_classifier,
    EFFECTIVENESS_ORDER,
    SIDE_EFFECTS_ORDER,
    decode_effectiveness,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("/app/models/global_best_model.pkl")

app = FastAPI(
    title="Drug Effectiveness Prediction API",
    description="FastAPI service for predicting drug effectiveness from patient reviews",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
def load_model(path: Path):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    m = joblib.load(path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(m).__name__}")
    if hasattr(m, "named_steps"):
        print(f"  Pipeline steps: {list(m.named_steps.keys())}")
    return m


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"✗ ERROR: Failed to load model from {MODEL_PATH}")
    print(f"  Error: {e}")
    model = None


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class DrugReviewInput(BaseModel):
    """Single drug review input for prediction."""
    urlDrugName: str = Field(..., description="Name of the drug")
    condition: str = Field(..., description="Medical condition being treated")
    benefitsReview: str = Field("", description="Review text about benefits")
    sideEffectsReview: str = Field("", description="Review text about side effects")
    commentsReview: str = Field("", description="Additional comments")
    rating: float = Field(..., ge=1, le=10, description="Overall rating (1-10)")
    sideEffects: str = Field(
        "Moderate Side Effects",
        description="Side effects severity category"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "urlDrugName": "Lipitor",
                "condition": "High Cholesterol",
                "benefitsReview": "This medication helped lower my cholesterol significantly.",
                "sideEffectsReview": "Some minor muscle aches but manageable.",
                "commentsReview": "Overall satisfied with the results.",
                "rating": 8.0,
                "sideEffects": "Mild Side Effects",
            }
        }


class PredictRequest(BaseModel):
    """
    Prediction request with list of instances (dicts of features).
    """
    instances: List[Dict[str, Any]] = Field(
        ..., description="List of drug review instances to predict"
    )


class PredictionResponse(BaseModel):
    """Response containing predictions."""
    predictions: List[str] = Field(..., description="Predicted effectiveness labels")
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None, description="Class probabilities for each prediction"
    )


# -----------------------------------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# -----------------------------------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    """
    Predict drug effectiveness from review data.
    
    Accepts a list of instances, each containing:
    - urlDrugName: Drug name
    - condition: Medical condition
    - benefitsReview: Benefits review text
    - sideEffectsReview: Side effects review text
    - commentsReview: Additional comments
    - rating: Overall rating (1-10)
    - sideEffects: Side effects severity category
    
    Returns predicted effectiveness labels and optional probabilities.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    instances = request.instances
    if not instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided for prediction.",
        )

    try:
        # Convert to DataFrame
        df = pd.DataFrame(instances)
        
        # Ensure required columns exist
        required_cols = [
            "urlDrugName", "condition", "benefitsReview",
            "sideEffectsReview", "commentsReview", "rating", "sideEffects"
        ]
        for col in required_cols:
            if col not in df.columns:
                if col in ["benefitsReview", "sideEffectsReview", "commentsReview"]:
                    df[col] = ""
                elif col == "sideEffects":
                    df[col] = "Moderate Side Effects"
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required field: {col}",
                    )

        # Make predictions
        predictions_encoded = model.predict(df)
        predictions = [decode_effectiveness(p) for p in predictions_encoded]

        # Get probabilities if available
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                probabilities = [
                    {EFFECTIVENESS_ORDER[i]: float(p[i]) for i in range(len(EFFECTIVENESS_ORDER))}
                    for p in proba
                ]
            except Exception:
                pass  # Some models don't support predict_proba

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# -----------------------------------------------------------------------------
# Model Info Endpoint
# -----------------------------------------------------------------------------
@app.get("/model-info")
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return {"error": "Model not loaded"}
    
    info = {
        "model_type": type(model).__name__,
        "effectiveness_classes": EFFECTIVENESS_ORDER,
        "side_effects_categories": SIDE_EFFECTS_ORDER,
    }
    
    if hasattr(model, "named_steps"):
        info["pipeline_steps"] = list(model.named_steps.keys())
    
    return info


# -----------------------------------------------------------------------------
# Single Prediction Endpoint (Convenience)
# -----------------------------------------------------------------------------
@app.post("/predict-single")
def predict_single(review: DrugReviewInput):
    """
    Predict effectiveness for a single drug review.
    Convenience endpoint that accepts a single review object.
    """
    request = PredictRequest(instances=[review.model_dump()])
    response = predict(request)
    
    return {
        "prediction": response.predictions[0],
        "probabilities": response.probabilities[0] if response.probabilities else None,
    }
