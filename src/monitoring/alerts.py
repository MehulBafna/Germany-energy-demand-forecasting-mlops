"""
alerts.py — Read drift reports and trigger automated retraining.

This is the self-healing layer of the MLOps pipeline.
Checks drift status and decides whether to retrain, alert, or do nothing.

Alert levels:
  INFO    — stable, everything fine
  WARNING — moderate drift, monitoring started
  CRITICAL— significant drift, retraining triggered automatically
"""

import json
import logging
import smtplib
import os
import yaml
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MONITOR_PARAMS = params["monitoring"]
REPORTS_PATH   = Path(MONITOR_PARAMS["reports_path"])

# Alert history — prevents spamming same alert repeatedly
ALERT_HISTORY_PATH = Path("monitoring/alert_history.json")


# ── Alert History ──────────────────────────────────────────────────────

def load_alert_history() -> dict:
    """Load alert history to avoid duplicate notifications."""
    if ALERT_HISTORY_PATH.exists():
        with open(ALERT_HISTORY_PATH) as f:
            return json.load(f)
    return {"alerts": [], "last_retrain": None}


def save_alert_history(history: dict):
    ALERT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def already_alerted_today(history: dict, alert_type: str) -> bool:
    """Check if we already sent this alert type today."""
    today = datetime.now().strftime("%Y-%m-%d")
    return any(
        a["date"] == today and a["type"] == alert_type
        for a in history["alerts"][-10:]  # Check last 10 alerts
    )


# ── Notifications ──────────────────────────────────────────────────────

def send_email_alert(subject: str, body: str):
    """
    Send email alert when drift is detected or retraining completes.
    Configure SMTP settings in .env file.
    Optional — system works without email too.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    alert_to  = os.getenv("ALERT_EMAIL")

    if not all([smtp_host, smtp_user, smtp_pass, alert_to]):
        logger.info("Email not configured — skipping notification (set SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL in .env)")
        return

    try:
        msg = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = alert_to
        msg["Subject"] = f"[EnergyPulse] {subject}"
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL(smtp_host, 465) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        logger.info(f"Email alert sent to {alert_to}: {subject}")
    except Exception as e:
        logger.warning(f"Email alert failed: {e}")


def send_slack_alert(message: str):
    """
    Send Slack notification via webhook.
    Optional — set SLACK_WEBHOOK_URL in .env
    """
    import requests
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.info("Slack not configured — skipping (set SLACK_WEBHOOK_URL in .env)")
        return

    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=5)
        if resp.status_code == 200:
            logger.info("Slack alert sent")
        else:
            logger.warning(f"Slack alert failed: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Slack alert failed: {e}")


def notify(subject: str, body: str, level: str = "INFO"):
    """Send all configured notifications."""
    emoji = {"INFO": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level, "📢")
    full_message = f"{emoji} *EnergyPulse Alert* | {subject}\n\n{body}"

    send_email_alert(subject, body)
    send_slack_alert(full_message)


# ── Core Alert Logic ───────────────────────────────────────────────────

def read_latest_drift_report() -> dict:
    """Load the most recent drift report."""
    latest_path = REPORTS_PATH / "drift_report_latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No drift report found at {latest_path}. Run drift.py first."
        )
    with open(latest_path) as f:
        return json.load(f)


def trigger_retraining():
    """
    Trigger the retraining pipeline.
    Imports and runs the retrain pipeline directly.
    In production this would call a Prefect/Airflow flow.
    """
    logger.info("🔄 Triggering automated retraining pipeline...")
    try:
        from pipelines.retrain_pipeline import run_retrain
        success, metrics = run_retrain()

        if success:
            logger.info(f"✅ Retraining complete. New MAE: {metrics.get('mae', 'N/A'):,.0f} MW")
            return True, metrics
        else:
            logger.error("❌ Retraining failed")
            return False, {}

    except Exception as e:
        logger.error(f"Retraining pipeline error: {e}")
        return False, {}


def process_alert(report: dict) -> str:
    """
    Core decision logic.
    Reads drift report and takes appropriate action.

    Returns: action taken ("no_action" / "monitor" / "retrained")
    """
    history = load_alert_history()
    status  = report["overall_status"]
    ts      = report["timestamp"]

    logger.info(f"Processing drift report from {ts}")
    logger.info(f"Status: {status.upper()} | Recommendation: {report['recommendation']}")

    # ── Stable — no action ──
    if status == "stable":
        logger.info("✅ All good — no drift detected")
        history["alerts"].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "type": "stable",
            "status": status,
        })
        save_alert_history(history)
        return "no_action"

    # ── Moderate drift — warn but don't retrain yet ──
    if status == "moderate_drift":
        if not already_alerted_today(history, "moderate_drift"):
            msg = (
                f"Moderate data drift detected in EnergyPulse.\n\n"
                f"Target PSI     : {report['target_drift']['psi']:.4f}\n"
                f"Mean shift     : {report['target_drift']['mean_shift_pct']:.1f}%\n"
                f"Drifted features: {report['feature_drift']['n_features_drifted']} / "
                f"{report['feature_drift']['n_features_checked']}\n\n"
                f"No action taken yet. Will retrain if drift worsens.\n"
                f"Check dashboard for details."
            )
            notify("Moderate Drift Detected", msg, level="WARNING")
            logger.warning("⚠️ Moderate drift — monitoring, no retraining yet")

            history["alerts"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "type": "moderate_drift",
                "target_psi": report["target_drift"]["psi"],
            })
            save_alert_history(history)
        else:
            logger.info("Moderate drift already alerted today — skipping duplicate")
        return "monitor"

    # ── Significant drift — trigger retraining ──
    if status == "significant_drift":
        if not MONITOR_PARAMS.get("retrain_on_drift", True):
            logger.warning("🔴 Significant drift detected but auto-retrain is disabled in params.yaml")
            notify(
                "Significant Drift — Manual Retraining Required",
                f"Significant drift detected. Auto-retrain is disabled.\n"
                f"Please retrain manually: python -m pipelines.retrain_pipeline",
                level="CRITICAL"
            )
            return "monitor"

        logger.warning("🔴 Significant drift — triggering automated retraining...")

        # Notify start of retraining
        notify(
            "Retraining Triggered by Drift",
            f"Significant drift detected. Automated retraining started.\n\n"
            f"Target PSI      : {report['target_drift']['psi']:.4f}\n"
            f"Mean shift      : {report['target_drift']['mean_shift_pct']:.1f}%\n"
            f"Drifted features: {report['feature_drift']['n_features_drifted']}",
            level="CRITICAL"
        )

        # Run retraining
        success, metrics = trigger_retraining()

        if success:
            promoted = metrics.get("promoted_to_production", False)
            notify(
                "Retraining Complete",
                f"Retraining finished successfully.\n\n"
                f"New MAE    : {metrics.get('mae', 'N/A'):,.0f} MW\n"
                f"Promoted   : {'Yes ✅' if promoted else 'No — champion retained'}\n"
                f"Drift resolved: monitoring continues",
                level="INFO"
            )
        else:
            notify(
                "Retraining Failed",
                "Automated retraining failed. Please check logs and retrain manually.",
                level="CRITICAL"
            )

        history["last_retrain"] = datetime.now().isoformat()
        history["alerts"].append({
            "date":    datetime.now().strftime("%Y-%m-%d"),
            "type":    "significant_drift_retrain",
            "success": success,
            "metrics": metrics,
        })
        save_alert_history(history)
        return "retrained"

    return "no_action"


def run_alert_check() -> str:
    """
    Main entry point.
    Read latest drift report and process alerts.
    Called by the daily Prefect scheduler after drift check.
    """
    logger.info("Running alert check...")
    try:
        report = read_latest_drift_report()
        action = process_alert(report)
        logger.info(f"Alert check complete. Action taken: {action}")
        return action
    except FileNotFoundError as e:
        logger.error(f"Alert check failed: {e}")
        return "error"


if __name__ == "__main__":
    action = run_alert_check()
    print(f"\nAction taken: {action}")
