"""
evaluate.py — Detailed model evaluation with visualizations.

Generates:
  1. Predictions vs Actuals plot
  2. Error distribution plot
  3. Performance by hour of day
  4. Performance by month (seasonal analysis)
  5. Worst prediction days
  6. Full evaluation report (JSON)

All plots + report are logged to MLflow automatically.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

DATA_PARAMS   = params["data"]
FEAT_PARAMS   = params["features"]
TRAIN_PARAMS  = params["training"]
MLFLOW_PARAMS = params["mlflow"]

PLOTS_PATH = Path("monitoring/plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)


# ── Plotting Helpers ───────────────────────────────────────────────────

def plot_predictions_vs_actuals(df_eval: pd.DataFrame, save_path: Path) -> Path:
    """
    Line chart: actual load vs predicted load over time.
    Shows last 14 days for clarity.
    """
    df_plot = df_eval.tail(24 * 14)  # Last 14 days

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Top: full 14-day view
    ax1 = axes[0]
    ax1.plot(df_plot["timestamp"], df_plot["actual"],
             label="Actual", color="#2196f3", linewidth=1.5, alpha=0.9)
    ax1.plot(df_plot["timestamp"], df_plot["predicted"],
             label="Predicted", color="#f44336", linewidth=1.5, alpha=0.8, linestyle="--")
    ax1.set_title("Actual vs Predicted Energy Load — Last 14 Days", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Load (MW)")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.grid(True, alpha=0.3)

    # Bottom: error over time
    ax2 = axes[1]
    error = df_plot["actual"] - df_plot["predicted"]
    ax2.fill_between(df_plot["timestamp"], error, 0,
                     where=(error >= 0), color="#4caf50", alpha=0.5, label="Under-predicted")
    ax2.fill_between(df_plot["timestamp"], error, 0,
                     where=(error < 0),  color="#f44336", alpha=0.5, label="Over-predicted")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Prediction Error (Actual − Predicted)", fontsize=12)
    ax2.set_ylabel("Error (MW)")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_error_distribution(df_eval: pd.DataFrame, save_path: Path) -> Path:
    """
    Histogram of prediction errors.
    Good model = errors centered around 0, roughly bell-shaped.
    Skewed distribution = systematic bias.
    """
    errors = df_eval["actual"] - df_eval["predicted"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=60, color="#667eea", edgecolor="white", alpha=0.8)
    ax1.axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
    ax1.axvline(errors.mean(), color="orange", linewidth=2,
                linestyle="-", label=f"Mean: {errors.mean():,.0f} MW")
    ax1.set_title("Error Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Error (MW)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot by day of week
    ax2 = axes[1]
    df_eval["error"] = errors
    df_eval["day_name"] = pd.to_datetime(df_eval["timestamp"]).dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_errors = [df_eval[df_eval["day_name"] == d]["error"].values for d in day_order]
    bp = ax2.boxplot(day_errors, tick_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#667eea")
        patch.set_alpha(0.7)
    ax2.axhline(0, color="red", linewidth=1, linestyle="--")
    ax2.set_title("Error by Day of Week", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Error (MW)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_performance_by_hour(df_eval: pd.DataFrame, save_path: Path) -> Path:
    """
    MAE by hour of day.
    Reveals when the model struggles most.
    Common finding: model is worse at peak hours (9am, 8pm)
    and better at stable night hours (2-4am).
    """
    df_eval = df_eval.copy()
    df_eval["hour"] = pd.to_datetime(df_eval["timestamp"]).dt.hour
    df_eval["abs_error"] = (df_eval["actual"] - df_eval["predicted"]).abs()

    hourly_mae = df_eval.groupby("hour")["abs_error"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(hourly_mae.index, hourly_mae.values,
                  color=["#f44336" if v > hourly_mae.mean() * 1.2 else "#4caf50"
                         for v in hourly_mae.values],
                  edgecolor="white", alpha=0.85)
    ax.axhline(hourly_mae.mean(), color="navy", linewidth=2,
               linestyle="--", label=f"Overall MAE: {hourly_mae.mean():,.0f} MW")
    ax.set_title("MAE by Hour of Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Absolute Error (MW)")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    # Red bars = worse than average, green = better
    ax.text(0.02, 0.95, "Red = worse than average  |  Green = better than average",
            transform=ax.transAxes, fontsize=9, va="top", color="dimgray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_performance_by_month(df_eval: pd.DataFrame, save_path: Path) -> Path:
    """
    MAE by month — reveals seasonal weaknesses.
    Model often struggles more in transition months (Mar, Oct)
    when patterns shift from summer to winter mode.
    """
    df_eval = df_eval.copy()
    df_eval["month"] = pd.to_datetime(df_eval["timestamp"]).dt.month
    df_eval["abs_error"] = (df_eval["actual"] - df_eval["predicted"]).abs()

    month_mae = df_eval.groupby("month")["abs_error"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 5))
    available_months = sorted(month_mae.index.tolist())
    available_names = [month_names[m-1] for m in available_months]
    available_values = [month_mae[m] for m in available_months]

    ax.bar(range(1, len(available_months) + 1),
       available_values,
       color="#667eea", edgecolor="white", alpha=0.85)
    ax.axhline(month_mae.mean(), color="red", linewidth=2,
           linestyle="--", label=f"Average: {month_mae.mean():,.0f} MW")
    ax.set_title("MAE by Month (Seasonal Analysis)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Absolute Error (MW)")
    ax.set_xticks(range(1, len(available_months) + 1))
    ax.set_xticklabels(available_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")
    return save_path


# ── Main Evaluation ────────────────────────────────────────────────────

def evaluate(run_id: str = None) -> dict:
    """
    Full evaluation pipeline.
    Loads model + test data, generates all plots, logs to MLflow.
    """
    # Load model and data
    model_path = Path(TRAIN_PARAMS["model_output_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    model = joblib.load(model_path)

    processed_path = Path(DATA_PARAMS["processed_path"])
    df = pd.read_csv(processed_path, parse_dates=["timestamp"])

    target       = FEAT_PARAMS["target"]
    feat_path    = Path("models/feature_columns.json")
    with open(feat_path) as f:
        feature_cols = json.load(f)

    # Use test set only
    test_days   = DATA_PARAMS["test_split_days"]
    split_point = df["timestamp"].max() - pd.Timedelta(days=test_days)
    test_df     = df[df["timestamp"] > split_point].copy()

    X_test = test_df[feature_cols]
    y_test = test_df[target].values
    y_pred = model.predict(X_test)

    df_eval = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "actual":    y_test,
        "predicted": y_pred,
    })

    # Generate all plots
    logger.info("Generating evaluation plots...")
    plot_paths = [
        plot_predictions_vs_actuals(df_eval.copy(), PLOTS_PATH / "predictions_vs_actuals.png"),
        plot_error_distribution(df_eval.copy(),     PLOTS_PATH / "error_distribution.png"),
        plot_performance_by_hour(df_eval.copy(),    PLOTS_PATH / "performance_by_hour.png"),
        plot_performance_by_month(df_eval.copy(),   PLOTS_PATH / "performance_by_month.png"),
    ]

    # Compute final metrics
    errors     = np.abs(df_eval["actual"] - df_eval["predicted"])
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "test_period_days": test_days,
        "n_test_samples": len(df_eval),
        "mae_mw":    round(errors.mean(), 2),
        "rmse_mw":   round(np.sqrt((errors**2).mean()), 2),
        "mape_pct":  round((errors / df_eval["actual"]).mean() * 100, 4),
        "p95_error": round(np.percentile(errors, 95), 2),  # 95th percentile error
        "worst_day": str(df_eval.loc[errors.idxmax(), "timestamp"]),
        "best_day":  str(df_eval.loc[errors.idxmin(), "timestamp"]),
    }

    # Save report
    report_path = Path("monitoring/evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n{'='*45}")
    logger.info(f"Evaluation Report")
    logger.info(f"{'='*45}")
    logger.info(f"MAE        : {report['mae_mw']:,.0f} MW")
    logger.info(f"RMSE       : {report['rmse_mw']:,.0f} MW")
    logger.info(f"MAPE       : {report['mape_pct']:.2f}%")
    logger.info(f"P95 Error  : {report['p95_error']:,.0f} MW")
    logger.info(f"Worst hour : {report['worst_day']}")

    # Log to MLflow if run_id provided
    if run_id:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_PARAMS["tracking_uri"]))
        with mlflow.start_run(run_id=run_id):
            for path in plot_paths:
                mlflow.log_artifact(str(path), "evaluation_plots")
            mlflow.log_artifact(str(report_path))
            mlflow.log_metric("p95_error", report["p95_error"])

    return report


if __name__ == "__main__":
    report = evaluate()
    print(f"\nPlots saved to: {PLOTS_PATH}")
    print(f"Report saved to: monitoring/evaluation_report.json")
