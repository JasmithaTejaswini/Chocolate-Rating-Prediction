from pathlib import Path
import time

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List


# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# Create FastAPI application
app = FastAPI(
    title="Chocolate Rating Prediction API",
    description="ML API for predicting chocolate bar ratings",
    version="1.0.0"
)


# Input schema
class ChocolateInput(BaseModel):
    company: str
    specific_bean_origin_or_bar_name: str
    review_date: float
    cocoa_percent: float = Field(gt=0, le=100)
    company_location: str
    bean_type: str = "Unknown"
    broad_bean_origin: str


# Output schema
class PredictionResponse(BaseModel):
    predicted_rating: float
    model_name: str
    latency_ms: float


# Convert API input into the exact columns expected by the model
def prepare_input(data):

    records = []

    for item in data:
        records.append({
            "Company\xa0\n(Maker-if known)": item.company,
            "Specific Bean Origin\nor Bar Name":
                item.specific_bean_origin_or_bar_name,
            "Review\nDate": item.review_date,
            "Cocoa\nPercent": item.cocoa_percent,
            "Company\nLocation": item.company_location,
            "Bean\nType": item.bean_type,
            "Broad Bean\nOrigin": item.broad_bean_origin
        })

    return pd.DataFrame(records)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Chocolate Rating API is running"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Model metadata endpoint
@app.get("/metadata")
def metadata():
    return {
        "model_name": "Random Forest",
        "model_file": "best_model.joblib",
        "target": "Rating",
        "framework": "Scikit-learn"
    }


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: ChocolateInput):

    try:
        start_time = time.perf_counter()

        input_data = prepare_input([data])

        prediction = model.predict(input_data)[0]

        latency = (time.perf_counter() - start_time) * 1000

        return {
            "predicted_rating": round(float(prediction), 3),
            "model_name": "Random Forest",
            "latency_ms": round(latency, 3)
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(error)}"
        )


# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(data: List[ChocolateInput]):

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Input batch cannot be empty"
        )

    try:
        start_time = time.perf_counter()

        input_data = prepare_input(data)

        predictions = model.predict(input_data)

        latency = (time.perf_counter() - start_time) * 1000

        return {
            "predictions": [
                round(float(value), 3) for value in predictions
            ],
            "total_records": len(data),
            "latency_ms": round(latency, 3)
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: {str(error)}"
        )