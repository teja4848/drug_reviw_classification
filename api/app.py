# api/app.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Imports for class labels
# -----------------------------------------------------------------------------
try:
    from drug_pipeline import EFFECTIVENESS_CLASSES
except Exception:
    try:
        from api.drug_pipeline import EFFECTIVENESS_CLASSES
    except Exception:
        from .drug_pipeline import EFFECTIVENESS_CLASSES


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
MODEL_CANDIDATES = [
    Path("/app/models/global_best_model.pkl"),
    Path("/app/models/global_best_model.joblib"),
    Path("models/global_best_model.pkl"),
    Path("models/global_best_model.joblib"),
    Path("models/global_best_model_optuna.pkl"),   # you have this in explorer
]

TFIDF_CANDIDATES = [
    Path("/app/models/tfidf_vectorizer.joblib"),
    Path("models/tfidf_vectorizer.joblib"),
]

app = FastAPI(title="Drug Effectiveness Prediction API", version="1.0.0")

model = None
tfidf = None
MODEL_PATH_USED: Optional[str] = None
TFIDF_PATH_USED: Optional[str] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def pick_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def decode_effectiveness(pred: Any) -> str:
    if pred is None:
        return "Unknown"
    if isinstance(pred, str):
        return pred
    try:
        idx = int(pred)
        if 0 <= idx < len(EFFECTIVENESS_CLASSES):
            return EFFECTIVENESS_CLASSES[idx]
    except Exception:
        pass
    return str(pred)


def normalize_text_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return ONLY the text fields (what the vectorizer needs).
    Accept both camelCase and snake_case.
    """
    df = df.copy()

    rename_map = {
        "benefitsReview": "benefits_review",
        "sideEffectsReview": "side_effects_review",
        "commentsReview": "comments_review",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    for col in ["benefits_review", "side_effects_review", "comments_review"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    return df[["benefits_review", "side_effects_review", "comments_review"]]


def combine_text(df_text: pd.DataFrame) -> pd.Series:
    return df_text.fillna("").astype(str).agg(" ".join, axis=1)


def predict_with_fallback(df_text: pd.DataFrame):
    """
    1) If model is a Pipeline that can handle DataFrame -> model.predict(df_text)
    2) Else if tfidf is available -> tfidf.transform(combined_text) -> model.predict(X)
    """
    global model, tfidf

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Try pipeline first (if it exists)
    try:
        # Many pipelines accept DataFrame; if it works, use it
        preds = model.predict(df_text)
        proba = model.predict_proba(df_text) if hasattr(model, "predict_proba") else None
        return preds, proba
    except Exception:
        pass

    # Fallback: model expects numeric features, so use TF-IDF
    if tfidf is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model appears to require TF-IDF features, but tfidf_vectorizer.joblib "
                "was not found/loaded. Please ensure it exists in models/."
            ),
        )

    combined = combine_text(df_text)
    X = tfidf.transform(combined)

    preds = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return preds, proba


# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    global model, tfidf, MODEL_PATH_USED, TFIDF_PATH_USED

    mp = pick_existing(MODEL_CANDIDATES)
    if mp is None:
        print("✗ No model file found in:", [str(p) for p in MODEL_CANDIDATES])
        model = None
    else:
        MODEL_PATH_USED = str(mp)
        print(f"Loading model: {MODEL_PATH_USED}")
        model = joblib.load(mp)
        print("✓ Model loaded:", type(model).__name__)

    tp = pick_existing(TFIDF_CANDIDATES)
    if tp is not None:
        TFIDF_PATH_USED = str(tp)
        print(f"Loading TF-IDF vectorizer: {TFIDF_PATH_USED}")
        tfidf = joblib.load(tp)
        print("✓ TF-IDF loaded:", type(tfidf).__name__)
    else:
        tfidf = None
        print("⚠️ TF-IDF vectorizer not found (ok only if model is full pipeline).")


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(...)


class PredictionResponse(BaseModel):
    predictions: List[str]
    probabilities: Optional[List[Dict[str, float]]] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tfidf_loaded": tfidf is not None,
        "model_path": MODEL_PATH_USED,
        "tfidf_path": TFIDF_PATH_USED,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__ if model is not None else None,
        "tfidf_type": type(tfidf).__name__ if tfidf is not None else None,
        "classes": EFFECTIVENESS_CLASSES,
        "model_path": MODEL_PATH_USED,
        "tfidf_path": TFIDF_PATH_USED,
        "pipeline_steps": list(getattr(model, "named_steps", {}).keys()) if hasattr(model, "named_steps") else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    if not req.instances:
        raise HTTPException(status_code=400, detail="No instances provided.")

    try:
        df_raw = pd.DataFrame(req.instances)
        df_text = normalize_text_only(df_raw)

        preds, proba = predict_with_fallback(df_text)
        labels = [decode_effectiveness(p) for p in preds]

        probs_out = None
        if proba is not None:
            probs_out = [
                {EFFECTIVENESS_CLASSES[i]: float(row[i]) for i in range(len(EFFECTIVENESS_CLASSES))}
                for row in proba
            ]

        return PredictionResponse(predictions=labels, probabilities=probs_out)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
