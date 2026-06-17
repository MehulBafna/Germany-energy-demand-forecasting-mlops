"""
drift.py — Data and target drift detection using Evidently AI.

Compares recent live data against training reference data.
Generates HTML drift report and triggers retraining if drift is significant.

Two types of drift monitored:
  1. Data drift   — feature distributions have shifted
  2. Target drift — actual load values have shifted (new baseline)

PSI thresholds (from params.yaml):
  < 0.10 → stable
  0.10–0.20 → moderate drift, monitor
  > 0.20 → significant drift, retrain
"""

import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MONITOR_PARAMS = params["monitoring"]
DATA_PARAMS    = params["data"]
FEAT_PARAMS    = params["features"]
REPORTS_PATH   = Path(MONITOR_PARAMS["reports_path"])
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# ── PSI Calculation ────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).

    Measures how much a distribution has shifted.
    Works by:
      1. Bucketing reference data into bins
      2. Measuring what % of current data falls in each bin
      3. Comparing the two % distributions

    PSI = Σ (current% - reference%) × ln(current% / reference%)

    Returns float: PSI score
    """
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    # Count how many values fall in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]

    # Convert to proportions, avoid division by zero
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)

    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def interpret_psi(psi: float) -> str:
    """Human-readable PSI interpretation."""
    if psi < 0.1:
        return "stable"
    elif psi < 0.2:
        return "moderate_drift"
    else:
        return "significant_drift"


# ── Drift Detection ────────────────────────────────────────────────────

class DriftDetector:
    """
    Monitors data and target drift between reference and current data.
    Reference = training data distribution
    Current   = recent live data (last N days)
    """

    def __init__(self):
        self.threshold = MONITOR_PARAMS["drift_threshold"]
        self.reference_stats = None

    def fit_reference(self, df: pd.DataFrame):
        """
        Compute and store reference statistics from training data.
        Called once after training. Saved to disk for future comparisons.
        """
        target = FEAT_PARAMS["target"]
        feat_cols = [c for c in df.columns if c not in ["timestamp", target]]

        self.reference_stats = {
            "target_mean":   float(df[target].mean()),
            "target_std":    float(df[target].std()),
            "target_min":    float(df[target].min()),
            "target_max":    float(df[target].max()),
            "target_values": df[target].values.tolist()[:5000],  # Sample for PSI
            "feature_stats": {},
            "fit_timestamp": datetime.now().isoformat(),
            "n_rows": len(df),
        }

        # Store distribution for each feature
        for col in feat_cols[:20]:  # Top 20 features to avoid bloat
            self.reference_stats["feature_stats"][col] = {
                "mean":   float(df[col].mean()),
                "std":    float(df[col].std()),
                "values": df[col].dropna().values.tolist()[:2000],
            }

        logger.info(f"Reference stats fitted on {len(df):,} rows")
        self._save_reference()

    def _save_reference(self):
        """Save reference stats to disk."""
        path = Path("models/reference_stats.json")
        with open(path, "w") as f:
            json.dump(self.reference_stats, f)
        logger.info(f"Reference stats saved to {path}")

    def load_reference(self):
        """Load reference stats from disk."""
        path = Path("models/reference_stats.json")
        if not path.exists():
            raise FileNotFoundError(
                "Reference stats not found. Run fit_reference() after training."
            )
        with open(path) as f:
            self.reference_stats = json.load(f)
        logger.info(f"Reference stats loaded (fitted: {self.reference_stats['fit_timestamp']})")

    def detect_drift(self, current_df: pd.DataFrame) -> dict:
        """
        Compare current data against reference distribution.

        Returns drift report with:
          - PSI score per feature
          - Target drift score
          - Overall drift status
          - Recommendation (retrain / monitor / stable)
        """
        if self.reference_stats is None:
            self.load_reference()

        target    = FEAT_PARAMS["target"]
        feat_cols = [c for c in current_df.columns
                     if c not in ["timestamp", target]
                     and c in self.reference_stats["feature_stats"]]

        logger.info(f"Detecting drift on {len(current_df):,} current rows "
                    f"vs {self.reference_stats['n_rows']:,} reference rows...")

        # ── Target drift ──
        ref_target  = np.array(self.reference_stats["target_values"])
        cur_target  = current_df[target].dropna().values
        target_psi  = compute_psi(ref_target, cur_target)
        target_status = interpret_psi(target_psi)

        # ── Feature drift ──
        feature_drift = {}
        drifted_features = []

        for col in feat_cols:
            ref_vals = np.array(self.reference_stats["feature_stats"][col]["values"])
            cur_vals = current_df[col].dropna().values
            if len(cur_vals) < 10:
                continue
            psi    = compute_psi(ref_vals, cur_vals)
            status = interpret_psi(psi)
            feature_drift[col] = {"psi": psi, "status": status}
            if psi >= self.threshold:
                drifted_features.append(col)

        # ── Mean shift check ──
        ref_mean = self.reference_stats["target_mean"]
        cur_mean = float(current_df[target].mean())
        mean_shift_pct = abs(cur_mean - ref_mean) / ref_mean * 100

        # ── Overall assessment ──
        n_drifted       = len(drifted_features)
        n_features      = len(feature_drift)
        drift_ratio     = n_drifted / n_features if n_features > 0 else 0

        if target_psi > self.threshold or drift_ratio > 0.3:
            overall_status  = "significant_drift"
            recommendation  = "RETRAIN"
        elif target_psi > 0.1 or drift_ratio > 0.15:
            overall_status  = "moderate_drift"
            recommendation  = "MONITOR"
        else:
            overall_status  = "stable"
            recommendation  = "NO_ACTION"

        report = {
            "timestamp":          datetime.now().isoformat(),
            "overall_status":     overall_status,
            "recommendation":     recommendation,
            "should_retrain":     recommendation == "RETRAIN",
            "target_drift": {
                "psi":            target_psi,
                "status":         target_status,
                "ref_mean_mw":    round(ref_mean, 2),
                "cur_mean_mw":    round(cur_mean, 2),
                "mean_shift_pct": round(mean_shift_pct, 2),
            },
            "feature_drift": {
                "n_features_checked": n_features,
                "n_features_drifted": n_drifted,
                "drift_ratio":        round(drift_ratio, 3),
                "drifted_features":   drifted_features[:10],
                "top_drifted": sorted(
                    feature_drift.items(),
                    key=lambda x: x[1]["psi"],
                    reverse=True
                )[:5],
            },
            "current_data_rows":  len(current_df),
            "drift_window_days":  MONITOR_PARAMS["drift_window_days"],
        }

        self._log_report(report)
        self._save_report(report)
        return report

    def _log_report(self, report: dict):
        """Log drift report to console."""
        status_emoji = {
            "stable":           "✅",
            "moderate_drift":   "⚠️",
            "significant_drift":"🔴",
        }
        emoji = status_emoji.get(report["overall_status"], "❓")

        logger.info(f"\n{'='*50}")
        logger.info(f"Drift Detection Report  {emoji}")
        logger.info(f"{'='*50}")
        logger.info(f"Overall Status  : {report['overall_status'].upper()}")
        logger.info(f"Recommendation  : {report['recommendation']}")
        logger.info(f"Target PSI      : {report['target_drift']['psi']:.4f}  "
                    f"({report['target_drift']['status']})")
        logger.info(f"Mean shift      : {report['target_drift']['mean_shift_pct']:.1f}%  "
                    f"({report['target_drift']['ref_mean_mw']:,.0f} → "
                    f"{report['target_drift']['cur_mean_mw']:,.0f} MW)")
        logger.info(f"Drifted features: {report['feature_drift']['n_features_drifted']} / "
                    f"{report['feature_drift']['n_features_checked']}")
        if report["feature_drift"]["drifted_features"]:
            logger.info(f"Top drifted     : {report['feature_drift']['drifted_features'][:3]}")
        logger.info(f"{'='*50}")

    def _save_report(self, report: dict):
        """Save drift report with timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        path = REPORTS_PATH / f"drift_report_{ts}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        # Also save as latest for easy access
        latest_path = REPORTS_PATH / "drift_report_latest.json"
        with open(latest_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Drift report saved: {path}")


def run_drift_check() -> dict:
    """
    Main entry point — fetch recent data, run drift detection.
    Called by the daily scheduler and the retrain pipeline.
    """
    from src.data.fetch import fetch_recent_data
    from src.data.preprocess import EnergyFeatureEngineer

    drift_days = MONITOR_PARAMS["drift_window_days"]
    logger.info(f"Fetching last {drift_days} days of data for drift check...")

    # Fetch recent live data
    recent_raw = fetch_recent_data(days_back=drift_days)

    # Apply same feature engineering as training
    engineer = EnergyFeatureEngineer.load()
    recent_features = engineer.transform(recent_raw)
    recent_features = recent_features.dropna()

    # Run drift detection
    detector = DriftDetector()
    report = detector.detect_drift(recent_features)

    return report


if __name__ == "__main__":
    report = run_drift_check()
    print(f"\nStatus      : {report['overall_status']}")
    print(f"Recommend   : {report['recommendation']}")
    print(f"Target PSI  : {report['target_drift']['psi']}")
    print(f"Should retrain: {report['should_retrain']}")
