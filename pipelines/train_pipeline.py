"""
train_pipeline.py — Full training pipeline orchestrator.

Runs all stages in order:
  1. Fetch fresh data from ENTSO-E
  2. Preprocess + feature engineering
  3. Train LightGBM model
  4. Evaluate and generate reports

Can be run:
  - Directly:  python -m pipelines.train_pipeline
  - Via DVC:   dvc repro
  - Via CI/CD: called by GitHub Actions
  - Via Prefect scheduler: called by daily_refresh.py
"""

import logging
import yaml
import json
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)


def run_pipeline(force_fetch: bool = False) -> tuple[bool, dict]:
    """
    Run full training pipeline end to end.

    Args:
        force_fetch: If True, fetch fresh data even if recent data exists

    Returns:
        (success: bool, metrics: dict)
    """
    pipeline_start = datetime.now()
    logger.info("=" * 55)
    logger.info("EnergyPulse Training Pipeline Starting")
    logger.info(f"Time: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 55)

    results = {
        "pipeline_start": pipeline_start.isoformat(),
        "stages": {},
        "success": False,
        "final_metrics": {},
    }

    # ── Stage 1: Fetch Data ────────────────────────────────────
    logger.info("\n[1/4] Fetching energy data...")
    try:
        from src.data.fetch import run_fetch
        df_raw = run_fetch()
        results["stages"]["fetch"] = {
            "status": "success",
            "rows": len(df_raw),
            "date_range": f"{df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}"
        }
        logger.info(f"✓ Fetch complete: {len(df_raw):,} rows")
    except Exception as e:
        logger.error(f"✗ Fetch failed: {e}")
        results["stages"]["fetch"] = {"status": "failed", "error": str(e)}
        return False, results

    # ── Stage 2: Preprocess ────────────────────────────────────
    logger.info("\n[2/4] Feature engineering...")
    try:
        from src.data.preprocess import run_preprocessing
        df_features, engineer = run_preprocessing()
        results["stages"]["preprocess"] = {
            "status": "success",
            "rows": len(df_features),
            "features": len(engineer.feature_columns),
        }
        logger.info(f"✓ Preprocessing complete: {len(engineer.feature_columns)} features")
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        results["stages"]["preprocess"] = {"status": "failed", "error": str(e)}
        return False, results

    # ── Stage 3: Train ─────────────────────────────────────────
    logger.info("\n[3/4] Training model...")
    try:
        from src.models.train import train
        model, metrics, run_id = train(df_features)
        results["stages"]["train"] = {
            "status": "success",
            "mlflow_run_id": run_id,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
        }
        logger.info(f"✓ Training complete: MAE={metrics['mae']:,.0f} MW")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        results["stages"]["train"] = {"status": "failed", "error": str(e)}
        return False, results

    # ── Stage 4: Evaluate ──────────────────────────────────────
    logger.info("\n[4/4] Evaluating model...")
    try:
        from src.models.evaluate import evaluate
        eval_report = evaluate(run_id=run_id)
        results["stages"]["evaluate"] = {
            "status": "success",
            "mae_mw": eval_report["mae_mw"],
            "mape_pct": eval_report["mape_pct"],
            "p95_error": eval_report["p95_error"],
        }
        logger.info(f"✓ Evaluation complete: MAE={eval_report['mae_mw']:,.0f} MW | "
                    f"MAPE={eval_report['mape_pct']:.2f}%")
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        results["stages"]["evaluate"] = {"status": "failed", "error": str(e)}
        # Evaluation failure doesn't stop deployment — model is still trained
        logger.warning("Continuing despite evaluation failure")

    # ── Also fit drift detector reference ─────────────────────
    logger.info("\n[+] Fitting drift detector reference...")
    try:
        from src.monitoring.drift import DriftDetector
        detector = DriftDetector()
        detector.fit_reference(df_features)
        logger.info("✓ Drift detector reference fitted")
    except Exception as e:
        logger.warning(f"Drift detector setup failed: {e}")

    # ── Pipeline complete ──────────────────────────────────────
    pipeline_end  = datetime.now()
    duration_mins = (pipeline_end - pipeline_start).total_seconds() / 60

    results["success"]         = True
    results["pipeline_end"]    = pipeline_end.isoformat()
    results["duration_minutes"]= round(duration_mins, 2)
    results["final_metrics"]   = metrics

    # Save pipeline run summary
    summary_path = Path("models/pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 55)
    logger.info("Pipeline Complete!")
    logger.info(f"Duration : {duration_mins:.1f} minutes")
    logger.info(f"MAE      : {metrics['mae']:,.0f} MW")
    logger.info(f"MAPE     : {metrics['mape']:.2f}%")
    logger.info("=" * 55)

    return True, results


if __name__ == "__main__":
    success, results = run_pipeline()
    sys.exit(0 if success else 1)
