"""
train.py — Train LightGBM forecasting model with MLflow tracking.

Every training run is logged to MLflow:
  - Parameters (from params.yaml)
  - Metrics (MAE, RMSE, MAPE)
  - Model artifact (saved to model registry)
  - Feature importance plot

Champion/Challenger pattern:
  - New model only replaces production model if MAE improves by threshold
  - Prevents accidentally deploying a worse model
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import yaml
import logging
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL_PARAMS   = params["model"]
TRAIN_PARAMS   = params["training"]
MLFLOW_PARAMS  = params["mlflow"]
DATA_PARAMS    = params["data"]
FEAT_PARAMS    = params["features"]


# ── Metrics ────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute forecasting metrics.

    MAE  — Mean Absolute Error (in MW — easy to interpret)
    RMSE — Root Mean Squared Error (penalises large errors more)
    MAPE — Mean Absolute Percentage Error (scale-independent %)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 4)}


# ── Time Series Cross Validation ───────────────────────────────────────

def time_series_cv(df: pd.DataFrame, feature_cols: list, target: str) -> dict:
    """
    Walk-forward cross validation for time series.

    Unlike random CV, time series CV always trains on past, tests on future.
    This mirrors real production conditions — you never know the future.

    Example with 3 folds:
      Fold 1: Train [Jan-Jun]  → Test [Jul]
      Fold 2: Train [Jan-Sep]  → Test [Oct]
      Fold 3: Train [Jan-Nov]  → Test [Dec]
    """
    n_splits = TRAIN_PARAMS["cv_folds"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    X = df[feature_cols].values
    y = df[target].values

    fold_metrics = []
    logger.info(f"Running {n_splits}-fold time series cross validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**MODEL_PARAMS["params"])
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(MODEL_PARAMS["early_stopping_rounds"], verbose=False)]
        )

        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        fold_metrics.append(metrics)
        logger.info(f"  Fold {fold}: MAE={metrics['mae']:,.0f} MW | "
                    f"RMSE={metrics['rmse']:,.0f} MW | MAPE={metrics['mape']:.2f}%")

    # Average across folds
    avg_metrics = {
        k: round(np.mean([m[k] for m in fold_metrics]), 2)
        for k in fold_metrics[0]
    }
    logger.info(f"CV Average → MAE={avg_metrics['mae']:,.0f} MW | "
                f"MAPE={avg_metrics['mape']:.2f}%")
    return avg_metrics


# ── Main Training ──────────────────────────────────────────────────────

def train(df: pd.DataFrame = None):
    """
    Full training run with MLflow logging.
    Loads processed features, trains model, logs to MLflow, saves if champion.
    """
    # ── Load data ──
    if df is None:
        processed_path = Path(DATA_PARAMS["processed_path"])
        if not processed_path.exists():
            raise FileNotFoundError(f"Run preprocess.py first: {processed_path}")
        df = pd.read_csv(processed_path, parse_dates=["timestamp"])

    target = FEAT_PARAMS["target"]
    feature_cols = [c for c in df.columns if c not in ["timestamp", target]]

    # ── Train / test split ──
    test_days   = DATA_PARAMS["test_split_days"]
    split_point = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["timestamp"] <= split_point]
    test_df  = df[df["timestamp"] >  split_point]

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target].values

    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    logger.info(f"Features: {len(feature_cols)}")

    # ── MLflow run ──
    mlflow.set_tracking_uri(MLFLOW_PARAMS["tracking_uri"])
    mlflow.set_experiment(MLFLOW_PARAMS["experiment_name"])

    with mlflow.start_run(run_name=f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Log all params from params.yaml
        mlflow.log_params(MODEL_PARAMS["params"])
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_test_rows", len(test_df))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("test_split_days", test_days)

        # ── Cross validation ──
        cv_metrics = time_series_cv(train_df, feature_cols, target)
        for k, v in cv_metrics.items():
            mlflow.log_metric(f"cv_{k}", v)

        # ── Final model training on full train set ──
        logger.info("Training final model on full training set...")
        model = lgb.LGBMRegressor(**MODEL_PARAMS["params"])
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(MODEL_PARAMS["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(100)
            ]
        )

        # ── Evaluate on test set ──
        y_pred = model.predict(X_test)
        test_metrics = compute_metrics(y_test, y_pred)
        logger.info(f"Test → MAE={test_metrics['mae']:,.0f} MW | "
                    f"RMSE={test_metrics['rmse']:,.0f} MW | "
                    f"MAPE={test_metrics['mape']:.2f}%")

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ── Feature importance ──
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        importance_path = Path("monitoring/feature_importance.csv")
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        logger.info("Top 10 features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']:<30} {row['importance']:>6.0f}")

        # ── Save model ──
        model_path = Path(TRAIN_PARAMS["model_output_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.lightgbm.log_model(model, "model")

        # Save feature column order — API needs this to align inputs
        feat_path = Path("models/feature_columns.json")
        with open(feat_path, "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact(str(feat_path))

        # ── Champion / Challenger ──
        promoted = _maybe_promote_to_production(
            run_id, test_metrics["mae"]
        )

        # ── Save run summary ──
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "cv_mae": cv_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_mape": test_metrics["mape"],
            "promoted_to_production": promoted,
            "n_features": len(feature_cols),
        }
        summary_path = Path("models/last_run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training complete. Promoted to production: {promoted}")
        return model, test_metrics, run_id


# ── Champion / Challenger ──────────────────────────────────────────────

def _maybe_promote_to_production(run_id: str, new_mae: float) -> bool:
    """
    Compare new model against current production champion.
    Only promote if new model is meaningfully better.

    This prevents:
      - Accidentally deploying a worse model
      - Noisy retraining from degrading production performance
    """
    client = mlflow.tracking.MlflowClient()
    model_name = MLFLOW_PARAMS["registered_model_name"]
    threshold  = MLFLOW_PARAMS["promote_if_mae_improvement"]

    try:
        # Get current production model MAE
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            # No production model yet — promote automatically
            logger.info("No production model exists — promoting new model automatically")
            _register_and_promote(client, run_id, model_name, stage="Production")
            return True

        prod_run_id = prod_versions[0].run_id
        prod_run    = client.get_run(prod_run_id)
        prod_mae    = prod_run.data.metrics.get("test_mae", float("inf"))

        improvement = (prod_mae - new_mae) / prod_mae

        logger.info(f"Champion MAE: {prod_mae:,.0f} MW | "
                    f"Challenger MAE: {new_mae:,.0f} MW | "
                    f"Improvement: {improvement:.2%}")

        if improvement >= threshold:
            logger.info(f"Challenger wins! Promoting to production...")
            _register_and_promote(client, run_id, model_name, stage="Production")
            # Archive old champion
            client.transition_model_version_stage(
                name=model_name,
                version=prod_versions[0].version,
                stage="Archived"
            )
            return True
        else:
            logger.info(f"Challenger did not beat champion by {threshold:.0%} — keeping champion")
            _register_and_promote(client, run_id, model_name, stage="Staging")
            return False

    except Exception as e:
        logger.warning(f"Model registry error: {e} — saving model locally only")
        return False


def _register_and_promote(client, run_id: str, model_name: str, stage: str):
    """Register model version and transition to given stage."""
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage
    )
    logger.info(f"Model v{mv.version} → {stage}")


if __name__ == "__main__":
    model, metrics, run_id = train()
    print(f"\n{'='*40}")
    print(f"Training Complete")
    print(f"{'='*40}")
    print(f"MAE  : {metrics['mae']:,.0f} MW")
    print(f"RMSE : {metrics['rmse']:,.0f} MW")
    print(f"MAPE : {metrics['mape']:.2f}%")
    print(f"Run ID: {run_id}")
    print(f"\nView in MLflow UI:")
    print(f"  mlflow ui --port 5000")
