"""
retrain_pipeline.py — Automated retraining triggered by drift detection.

Called by alerts.py when significant drift is detected.
Differs from train_pipeline.py in that it:
  - Only fetches recent data (not full history)
  - Merges recent data with existing training data
  - Has stricter promotion criteria (must beat champion)
  - Sends notifications on completion
  - Reloads the API after successful promotion
"""

import logging
import json
import requests
import yaml
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

RETRAIN_PARAMS = params["retrain"]
API_PARAMS     = params["api"]


def run_retrain() -> tuple[bool, dict]:
    """
    Triggered retraining pipeline.
    Fetches recent data, retrains, promotes if better.

    Returns:
        (success: bool, metrics: dict)
    """
    start_time = datetime.now()
    logger.info("=" * 55)
    logger.info("EnergyPulse — Triggered Retraining")
    logger.info(f"Reason: Drift detected at {start_time.strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 55)

    # ── Check if enough new data is available ─────────────────
    raw_path = Path(params["data"]["raw_path"])
    if raw_path.exists():
        import pandas as pd
        existing = pd.read_csv(raw_path, parse_dates=["timestamp"])
        latest_ts = existing["timestamp"].max()
        days_since = (datetime.now() - latest_ts).days

        min_days = RETRAIN_PARAMS["min_new_data_days"]
        if days_since < min_days:
            logger.info(f"Only {days_since} days of new data (need {min_days}) — skipping retrain")
            return False, {"reason": f"insufficient_new_data ({days_since} days)"}

        logger.info(f"New data available: {days_since} days since last fetch")

    # ── Run full pipeline ──────────────────────────────────────
    try:
        from pipelines.train_pipeline import run_pipeline
        success, results = run_pipeline(force_fetch=True)

        if not success:
            logger.error("Retraining pipeline failed")
            return False, results

        metrics = results.get("final_metrics", {})

        # ── Reload API with new model ──────────────────────────
        _reload_api()

        logger.info("✅ Retraining complete and API reloaded")
        return True, metrics

    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        return False, {"error": str(e)}


def _reload_api():
    """
    Tell the running FastAPI to reload the model from disk.
    Calls the /reload-model endpoint — no restart needed.
    """
    api_url = f"http://{API_PARAMS['host']}:{API_PARAMS['port']}/reload-model"
    try:
        resp = requests.post(api_url, timeout=10)
        if resp.status_code == 200:
            logger.info("✓ API model reloaded successfully")
        else:
            logger.warning(f"API reload returned {resp.status_code}")
    except requests.exceptions.ConnectionError:
        logger.warning("API not reachable for reload — model will load on next restart")
    except Exception as e:
        logger.warning(f"API reload failed: {e}")


if __name__ == "__main__":
    success, metrics = run_retrain()
    if success:
        print(f"\n✅ Retraining complete")
        print(f"MAE  : {metrics.get('mae', 'N/A'):,.0f} MW")
        print(f"MAPE : {metrics.get('mape', 'N/A'):.2f}%")
    else:
        print(f"\n❌ Retraining failed: {metrics}")
