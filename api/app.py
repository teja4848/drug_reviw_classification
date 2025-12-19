# api/app.py
"""
FastAPI service for housing price prediction.
Loads the trained model and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import shared pipeline components so unpickling works
from housing_pipeline import (
    ClusterSimilarity,
    column_ratio,
    ratio_name,
    build_preprocessing,
    make_estimator_for_name,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("/app/models/global_best_model_optuna.pkl")

app = FastAPI(
    title="Housing Price Prediction API",
    description="FastAPI service for predicting California housing prices",
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
    raise RuntimeError(f"Failed to load model: {e}")


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Prediction request with list of instances (dicts of features).
    """
    instances: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "longitude": -122.23,
                        "latitude": 37.88,
                        "housing_median_age": 41.0,
                        "total_rooms": 880.0,
                        "total_bedrooms": 129.0,
                        "population": 322.0,
                        "households": 126.0,
                        "median_income": 8.3252,
                        "ocean_proximity": "NEAR BAY",
                    }
                ]
            }
        }


class PredictResponse(BaseModel):
    predictions: List[float]
    count: int

    class Config:
        schema_extra = {
            "example": {
                "predictions": [452600.0],
                "count": 1,
            }
        }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Housing Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided. Please provide at least one instance.",
        )

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format. Could not convert to DataFrame: {e}",
        )

    required_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]
    missing = set(required_columns) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}",
        )

    preds_list = [float(p) for p in preds]

    return PredictResponse(predictions=preds_list, count=len(preds_list))


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Housing Price Prediction API - Starting Up")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")