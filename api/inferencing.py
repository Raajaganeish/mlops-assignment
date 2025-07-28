import json
from datetime import datetime, timedelta
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel, Field

from utils.db import get_prediction_stats, init_db, log_prediction

prediction_counter = Counter("prediction_requests_total", "Total prediction requests")

# Prometheus Gauges (defined globally)
total_requests_g = Gauge("total_requests", "Total prediction requests")
success_requests_g = Gauge("success_200", "Successful predictions (HTTP 200)")
bad_request_g = Gauge("bad_request_400", "Bad requests (HTTP 400)")
validation_errors_g = Gauge("validation_errors_422", "Validation errors (HTTP 422)")
internal_errors_g = Gauge("internal_errors_500", "Internal server errors (HTTP 500)")
avg_price_g = Gauge("avg_predicted_price", "Average predicted house price (USD)")
model_version_usage_g = Gauge(
    "model_version_usage", "Model version usage count", ["version"]
)

app = FastAPI(title="Housing Price Prediction API")


@app.on_event("startup")
def setup():
    init_db()


# Load model and scaler
model_package = joblib.load("models/best_model.pkl")
model = model_package["model"]
model_type = model_package.get("model_type", "UnknownModel")
model_version = model_package.get("model_version", "unknown")
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    log_prediction(
        input_json=body.decode("utf-8"),
        prediction_json=None,
        status_code=422,
        error_message=str(exc),
        model_type=model_type,
        model_version=model_version,
    )
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "details": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    body = await request.body()
    log_prediction(
        input_json=body.decode("utf-8"),
        prediction_json=None,
        status_code=exc.status_code,
        error_message=str(exc.detail),
        model_type=model_type,
        model_version=model_version,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTPException", "detail": exc.detail},
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


@app.get("/metrics")
def metrics(start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    try:
        start_dt = (
            datetime.fromisoformat(start)
            if start
            else datetime.utcnow() - timedelta(days=15)
        )
        end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid datetime format. Use ISO format."},
        )

    stats = get_prediction_stats(start=start_dt.isoformat(), end=end_dt.isoformat())

    total_requests_g.set(stats["total_requests"])
    success_requests_g.set(stats["success_200"])
    bad_request_g.set(stats["bad_request_400"])
    validation_errors_g.set(stats["validation_errors_422"])
    internal_errors_g.set(stats["internal_errors_500"])

    if stats["avg_predicted_price"] is not None:
        avg_price_g.set(stats["avg_predicted_price"])

    for version, count in stats["model_version_usage"].items():
        model_version_usage_g.labels(version=version).set(count)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics-json")
def metrics_json(start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    try:
        start_dt = (
            datetime.fromisoformat(start)
            if start
            else datetime.utcnow() - timedelta(days=15)
        )
        end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid datetime format. Use ISO format."},
        )

    return get_prediction_stats(start=start_dt.isoformat(), end=end_dt.isoformat())


@app.get("/")
def root():
    return {"message": "Welcome to the Housing Price Prediction API"}
