# ⚡ EnergyPulse — Automated Energy Demand Forecasting

> AI-powered German electricity demand forecasting with automated drift detection, retraining, and CI/CD deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange?logo=mlflow)
![DVC](https://img.shields.io/badge/Data-DVC-purple)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal?logo=fastapi)

---

## What It Does

EnergyPulse forecasts hourly electricity consumption for Germany's national grid (ENTSO-E) up to 48 hours ahead. Beyond the model itself, it implements a **production-grade MLOps system** that:

- Automatically detects when the model degrades (data drift)
- Triggers retraining without human intervention
- Deploys new models via CI/CD with zero downtime
- Monitors everything in a live Grafana dashboard

---

## Architecture

```
ENTSO-E API (Real German Grid Data)
        │
        ▼
┌───────────────────────────────────────────────────┐
│                  Data Pipeline                     │
│  fetch.py → preprocess.py (35 engineered features) │
│  Versioned with DVC                                │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│              Training Pipeline                     │
│  LightGBM + Time-Series CV                        │
│  MLflow experiment tracking + model registry       │
│  Champion/Challenger promotion pattern             │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│              FastAPI Service                       │
│  POST /predict  → 24-48h hourly forecast           │
│  GET  /drift    → drift status                     │
│  GET  /metrics  → live MAE/RMSE/MAPE               │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│           Monitoring & Self-Healing                │
│  Evidently AI → PSI drift detection daily          │
│  Auto-retraining when PSI > 0.15                   │
│  Grafana dashboard for live metrics                │
└───────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│                 CI/CD Pipeline                     │
│  GitHub Actions: test → build → deploy             │
│  Docker containerised, rolling restart             │
│  Weekly scheduled retraining (Sunday 3am)          │
└───────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Data Source | ENTSO-E API | Real German hourly energy data |
| Model | LightGBM | Gradient boosting forecaster |
| Experiment Tracking | MLflow | Log params, metrics, models |
| Data Versioning | DVC | Reproducible data + model versions |
| API | FastAPI | Serve predictions via REST |
| Containerisation | Docker | Consistent deployment |
| Orchestration | docker-compose | Run all services together |
| Drift Detection | Evidently AI | PSI-based distribution monitoring |
| Scheduling | Prefect | Daily drift check + retraining |
| CI/CD | GitHub Actions | Automated test, build, deploy |
| Monitoring | Grafana | Live metrics dashboard |

---

## Project Structure

```
energypulse/
├── src/
│   ├── data/
│   │   ├── fetch.py             # ENTSO-E data fetching
│   │   └── preprocess.py        # Feature engineering (35 features)
│   ├── models/
│   │   ├── train.py             # LightGBM + MLflow training
│   │   └── evaluate.py          # Metrics + evaluation plots
│   ├── monitoring/
│   │   ├── drift.py             # PSI drift detection
│   │   └── alerts.py            # Auto-retraining trigger
│   └── api/
│       └── app.py               # FastAPI endpoints
├── pipelines/
│   ├── train_pipeline.py        # Full pipeline orchestrator
│   └── retrain_pipeline.py      # Drift-triggered retraining
├── .github/workflows/
│   └── ci_cd.yml                # GitHub Actions CI/CD
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # All hyperparameters (DVC tracked)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/mehulbafna/energypulse
cd energypulse
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Add your ENTSOE_API_KEY (free at transparency.entsoe.eu)
```

### 3. Run full pipeline
```bash
# Option A — via DVC (recommended)
dvc repro

# Option B — directly
python -m pipelines.train_pipeline
```

### 4. Start all services
```bash
docker-compose up
```

### 5. Access services
| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3000 |

---

## API Usage

### Forecast next 24 hours
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"start_timestamp": "2024-06-15 14:00:00", "hours_ahead": 24}'
```

### Response
```json
{
  "forecast_start": "2024-06-15 14:00:00",
  "hours_ahead": 24,
  "predictions": [
    {
      "timestamp": "2024-06-15 14:00:00",
      "predicted_load_mw": 52340.5,
      "lower_bound_mw": 47106.5,
      "upper_bound_mw": 57574.6
    }
  ],
  "model_info": {"model_type": "LightGBM", "n_features": 35}
}
```

### Quick forecast from now
```bash
curl http://localhost:8000/predict/now?hours_ahead=12
```

### Check drift status
```bash
curl http://localhost:8000/drift
```

---

## MLOps Features

### Experiment Tracking
Every training run logged to MLflow with full reproducibility:
```bash
mlflow ui --port 5000
# Open localhost:5000 to compare experiments
```

### Data Versioning
```bash
dvc params diff     # What changed since last run?
dvc metrics show    # Compare metrics across versions
dvc dag             # Visualise pipeline
```

### Drift Detection
Runs daily at 2 AM. Compares last 7 days vs training distribution:
```bash
python -m src.monitoring.drift
# PSI < 0.10 → stable
# PSI 0.10-0.20 → moderate drift
# PSI > 0.20 → retraining triggered
```

### Manual retraining
```bash
python -m pipelines.retrain_pipeline
```

---

## Model Performance

| Metric | Value |
|---|---|
| MAE | ~1,200 MW |
| MAPE | ~2.5% |
| RMSE | ~1,700 MW |
| Forecast Horizon | 48 hours |
| Training Data | 2020-2024 (35,000+ hours) |

Top features: `lag_168h` (same hour last week), `lag_24h` (same hour yesterday), `hour_sin/cos`, `is_holiday`

---

## CI/CD Pipeline

Every push to `main` branch:
```
Push to GitHub
      ↓
pytest (unit + API tests)
      ↓  passes
Docker build + push to GHCR
      ↓  passes
Deploy to server (rolling restart)
      ↓
Health check passes → live
```

Weekly Sunday 3 AM:
```
Fetch fresh ENTSO-E data
      ↓
Retrain model
      ↓
Champion/Challenger evaluation
      ↓
Promote if MAE improves ≥ 2%
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ENTSOE_API_KEY` | Yes | ENTSO-E transparency platform token |
| `MLFLOW_TRACKING_URI` | No | Remote MLflow server (default: local) |
| `SLACK_WEBHOOK_URL` | No | Slack drift alerts |
| `ALERT_EMAIL` | No | Email drift notifications |
| `SERVER_HOST` | For deploy | Production server IP |
| `SSH_PRIVATE_KEY` | For deploy | SSH key for deployment |

---

## Data Source

Real hourly electricity load data from **ENTSO-E Transparency Platform** — the official European Network of Transmission System Operators for Electricity.

- Free API — register at [transparency.entsoe.eu](https://transparency.entsoe.eu)
- Germany (`DE`) data available from 2015
- Updates hourly

---

*Built by [Mehul Bafna](https://mehulbafna.github.io) | [LinkedIn](https://www.linkedin.com/in/mehul-bafna-8b696488/)*
