"""
app.py — FastAPI REST API for EnergyPulse predictions.

Endpoints:
  GET  /              — Health check
  GET  /health        — Detailed health + model info
  POST /predict       — Forecast energy load for next N hours
  GET  /predict/now   — Quick forecast from current time
  GET  /metrics       — Current model performance metrics
  GET  /drift         — Latest drift report summary
  POST /feedback      — Submit actual vs predicted for monitoring

Auto-generated API docs available at:
  http://localhost:8000/docs      (Swagger UI)
  http://localhost:8000/redoc     (ReDoc)
"""

import json
import logging
import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

API_PARAMS    = params["api"]
FEAT_PARAMS   = params["features"]
MONITOR_PARAMS= params["monitoring"]

# ── FastAPI App ────────────────────────────────────────────────────────

app = FastAPI(
    title="EnergyPulse API",
    description=(
        "AI-powered German electricity demand forecasting API. "
        "Predicts hourly energy load up to 48 hours ahead using "
        "LightGBM trained on ENTSO-E data with automated drift monitoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow cross-origin requests (needed for dashboards/frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model Loading ──────────────────────────────────────────────────────

class ModelStore:
    """
    Singleton that holds loaded model + feature engineer in memory.
    Loaded once at startup — not on every request (would be very slow).
    """
    model = None
    feature_engineer = None
    feature_columns = None
    loaded_at = None
    model_path = Path(params["training"]["model_output_path"])

    @classmethod
    def load(cls):
        """Load model and feature engineer from disk."""
        try:
            cls.model = joblib.load(cls.model_path)
            cls.feature_engineer = joblib.load(Path("models/feature_engineer.pkl"))
            with open("models/feature_columns.json") as f:
                cls.feature_columns = json.load(f)
            cls.loaded_at = datetime.now().isoformat()
            logger.info(f"Model loaded from {cls.model_path}")
            logger.info(f"Feature columns: {len(cls.feature_columns)}")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            logger.error("Run: python -m src.rag.embedder --init first")

    @classmethod
    def is_ready(cls) -> bool:
        return cls.model is not None and cls.feature_engineer is not None


# Load model when app starts
@app.on_event("startup")
async def startup_event():
    logger.info("Starting EnergyPulse API...")
    ModelStore.load()
    logger.info("API ready!")


# ── Pydantic Models (Request / Response schemas) ───────────────────────

class PredictRequest(BaseModel):
    """
    Input schema for /predict endpoint.
    Pydantic validates types automatically — wrong types = clear error.
    """
    start_timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:00:00"),
        description="Start time for forecast. Format: YYYY-MM-DD HH:00:00",
        example="2024-06-15 14:00:00"
    )
    hours_ahead: int = Field(
        default=24,
        ge=1,
        le=48,
        description="How many hours ahead to forecast (1-48)"
    )

class HourlyPrediction(BaseModel):
    timestamp: str
    predicted_load_mw: float
    lower_bound_mw: float   # Uncertainty estimate
    upper_bound_mw: float

class PredictResponse(BaseModel):
    forecast_start: str
    forecast_end: str
    hours_ahead: int
    predictions: List[HourlyPrediction]
    model_info: dict
    generated_at: str

class FeedbackRequest(BaseModel):
    """
    Submit actual observed values vs what was predicted.
    Used to track real-world model accuracy over time.
    """
    timestamp: str
    predicted_load_mw: float
    actual_load_mw: float
    notes: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loaded_at: Optional[str]
    uptime_seconds: float
    version: str


# ── Helper: Generate prediction features ──────────────────────────────

def build_prediction_features(start_ts: datetime, hours: int) -> pd.DataFrame:
    """
    Build feature matrix for future timestamps.

    Challenge: for future predictions we don't have actual lag values.
    Solution: use historical data as the "past" to compute lags,
              then predict forward hour by hour.
    """
    # Load recent historical data to compute lags
    raw_path = Path(params["data"]["raw_path"])
    if not raw_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Historical data not available. Run fetch.py first."
        )

    hist = pd.read_csv(raw_path, parse_dates=["timestamp"])
    hist = hist.sort_values("timestamp")

    # Create future timestamp rows with NaN load (to be predicted)
    future_ts = [start_ts + timedelta(hours=i) for i in range(hours)]
    future_df  = pd.DataFrame({
        "timestamp":       future_ts,
        "actual_load_mw":  np.nan
    })

    # Append future rows to history so lags can be computed
    combined = pd.concat([hist.tail(200), future_df], ignore_index=True)

    # Apply feature engineering
    featured = ModelStore.feature_engineer.transform(combined)

    # Return only the future rows
    future_features = featured[featured["timestamp"].isin(future_ts)].copy()

    # Fill any remaining NaN in features with forward fill
    future_features = future_features.fillna(method="ffill").fillna(0)

    return future_features


# ── Endpoints ──────────────────────────────────────────────────────────

_start_time = datetime.now()

@app.get("/", tags=["Health"])
async def root():
    """Simple health check."""
    return {
        "service": "EnergyPulse API",
        "status":  "running",
        "docs":    "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Detailed health check — model status, uptime."""
    uptime = (datetime.now() - _start_time).total_seconds()
    return HealthResponse(
        status        = "healthy" if ModelStore.is_ready() else "degraded",
        model_loaded  = ModelStore.is_ready(),
        model_loaded_at= ModelStore.loaded_at,
        uptime_seconds= round(uptime, 1),
        version       = "1.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
async def predict(request: PredictRequest):
    """
    Forecast German electricity load for the next N hours.

    Returns hourly predictions with uncertainty bounds.
    Uncertainty = ±10% of predicted value (simplified; extend with
    quantile regression for production-grade intervals).
    """
    if not ModelStore.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_ts = datetime.strptime(request.start_timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid timestamp format. Use: YYYY-MM-DD HH:MM:SS"
        )

    if request.hours_ahead > API_PARAMS["max_forecast_hours"]:
        raise HTTPException(
            status_code=400,
            detail=f"Max forecast horizon is {API_PARAMS['max_forecast_hours']} hours"
        )

    try:
        features = build_prediction_features(start_ts, request.hours_ahead)
        X = features[ModelStore.feature_columns].values
        raw_preds = ModelStore.model.predict(X)

        # Build response with uncertainty bounds
        predictions = []
        for i, (ts, pred) in enumerate(zip(features["timestamp"], raw_preds)):
            uncertainty = pred * 0.10  # ±10% simplified uncertainty
            predictions.append(HourlyPrediction(
                timestamp          = str(ts),
                predicted_load_mw  = round(float(pred), 1),
                lower_bound_mw     = round(float(pred - uncertainty), 1),
                upper_bound_mw     = round(float(pred + uncertainty), 1),
            ))

        end_ts = start_ts + timedelta(hours=request.hours_ahead - 1)

        return PredictResponse(
            forecast_start = str(start_ts),
            forecast_end   = str(end_ts),
            hours_ahead    = request.hours_ahead,
            predictions    = predictions,
            model_info     = {
                "model_type":   "LightGBM",
                "loaded_at":     ModelStore.loaded_at,
                "n_features":    len(ModelStore.feature_columns),
                "country":       params["data"]["country_code"],
            },
            generated_at = datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/now", tags=["Forecast"])
async def predict_now(hours_ahead: int = 24):
    """
    Shortcut endpoint — forecast from current time.
    No request body needed.

    Example: GET /predict/now?hours_ahead=12
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    request = PredictRequest(
        start_timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
        hours_ahead=hours_ahead
    )
    return await predict(request)


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Latest model performance metrics from evaluation report.
    Used by Grafana dashboard.
    """
    report_path = Path("monitoring/evaluation_report.json")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation report found. Run evaluate.py")

    with open(report_path) as f:
        return json.load(f)


@app.get("/drift", tags=["Monitoring"])
async def get_drift():
    """
    Latest drift detection report summary.
    Shows whether model needs retraining.
    """
    drift_path = Path(MONITOR_PARAMS["reports_path"]) / "drift_report_latest.json"
    if not drift_path.exists():
        raise HTTPException(status_code=404, detail="No drift report found. Run drift.py")

    with open(drift_path) as f:
        report = json.load(f)

    # Return summary (not full report)
    return {
        "status":          report["overall_status"],
        "recommendation":  report["recommendation"],
        "should_retrain":  report["should_retrain"],
        "target_psi":      report["target_drift"]["psi"],
        "mean_shift_pct":  report["target_drift"]["mean_shift_pct"],
        "checked_at":      report["timestamp"],
    }


@app.post("/feedback", tags=["Monitoring"])
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit actual observed load vs prediction.
    Logged for continuous accuracy tracking.
    Background task — doesn't slow down the response.
    """
    def save_feedback(fb: FeedbackRequest):
        feedback_path = Path("monitoring/feedback_log.jsonl")
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp":           fb.timestamp,
            "predicted_load_mw":   fb.predicted_load_mw,
            "actual_load_mw":      fb.actual_load_mw,
            "error_mw":            abs(fb.actual_load_mw - fb.predicted_load_mw),
            "error_pct":           abs(fb.actual_load_mw - fb.predicted_load_mw) / fb.actual_load_mw * 100,
            "logged_at":           datetime.now().isoformat(),
            "notes":               fb.notes,
        }
        with open(feedback_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    background_tasks.add_task(save_feedback, feedback)
    return {"status": "feedback logged", "timestamp": feedback.timestamp}


@app.post("/reload-model", tags=["Admin"])
async def reload_model():
    """
    Reload model from disk without restarting the API.
    Called automatically after successful retraining.
    """
    ModelStore.load()
    if ModelStore.is_ready():
        return {"status": "model reloaded", "loaded_at": ModelStore.loaded_at}
    else:
        raise HTTPException(status_code=500, detail="Model reload failed")


# ── Run directly ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=API_PARAMS["host"],
        port=API_PARAMS["port"],
        reload=True,   # Auto-reload on code changes (dev only)
        log_level="info"
    )
