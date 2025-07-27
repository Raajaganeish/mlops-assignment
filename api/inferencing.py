import joblib
import pandas as pd
import json
from typing import Optional
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from utils.db import (
    init_db,
    log_prediction,
    get_prediction_stats,
)
prediction_counter = Counter("prediction_requests_total", "Total prediction requests")

app = FastAPI(title="Housing Price Prediction API")


@app.on_event("startup")
def setup():
    init_db()


# Load model + metadata
model_package = joblib.load("models/best_model.pkl")
model = model_package["model"]
model_type = model_package.get("model_type", "UnknownModel")
model_version = model_package.get("model_version", "unknown")

# Load scaler
scaler = joblib.load("models/scaler.pkl")


# Input schema
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)


# Custom validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):

    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "details": exc.errors()},
    )


@app.post("/predict")
def predict(features: HousingFeatures):
    prediction_counter.inc()
    input_dict = features.model_dump()

    try:
        input_df = pd.DataFrame([input_dict])
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        prediction_dict = {
            "predicted_price": round(prediction * 100000, 2),
            "unit": "USD",
            "note": "Predicted median house value",
        }

        log_prediction(
            input_json=json.dumps(input_dict),
            prediction_json=json.dumps(prediction_dict),
            status_code=200,
            error_message=None,
            model_type=model_type,
            model_version=model_version,
        )

        return {
            **prediction_dict,
            "model_type": model_type,
            "model_version": model_version,
        }

    except Exception as e:
        log_prediction(
            input_json=json.dumps(input_dict),
            prediction_json=None,
            status_code=500,
            error_message=str(e),
            model_type=model_type,
            model_version=model_version,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction Failed", "details": str(e)},
        )


@app.get("/prediction-stats")
def prediction_stats(
    start: Optional[str] = Query(None, description="Start datetime (ISO format, UTC)"),
    end: Optional[str] = Query(None, description="End datetime (ISO format, UTC)"),
):
    stats = get_prediction_stats(start, end)
    return stats


 
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {"message": "Welcome to the Housing Price Prediction API"}
