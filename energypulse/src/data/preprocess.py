"""
preprocess.py — Feature engineering for energy consumption forecasting.

Transforms raw hourly MW readings into a rich feature set:
  - Calendar features   (hour, day, month, weekday, is_weekend)
  - German holidays     (binary flag — consumption drops on holidays)
  - Lag features        (load N hours ago — captures autocorrelation)
  - Rolling statistics  (rolling mean/std — captures trends and volatility)
  - Cyclical encoding   (sin/cos of hour/month — tells model 23h is close to 0h)

IMPORTANT: This same class is used in training AND in the API.
           Never preprocess differently in two places — training-serving skew
           is one of the most common silent bugs in production ML.
"""

import pandas as pd
import numpy as np
import holidays
import yaml
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

FEAT_PARAMS = params["features"]
DATA_PARAMS = params["data"]
PROCESSED_PATH = Path(DATA_PARAMS["processed_path"])
SCALER_PATH = Path("models/scaler.pkl")


class EnergyFeatureEngineer:
    """
    Transforms raw energy DataFrame into ML-ready feature matrix.
    Fit on training data, transform on any data (train/val/test/live).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        self.german_holidays = holidays.Germany()

    # ── Calendar Features ──────────────────────────────────────────────

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time components from timestamp."""
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
        df["day_of_month"]= df["timestamp"].dt.day
        df["month"]       = df["timestamp"].dt.month
        df["quarter"]     = df["timestamp"].dt.quarter
        df["year"]        = df["timestamp"].dt.year
        df["week_of_year"]= df["timestamp"].dt.isocalendar().week.astype(int)
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["is_monday"]   = (df["day_of_week"] == 0).astype(int)
        df["is_friday"]   = (df["day_of_week"] == 4).astype(int)
        return df

    def _add_cyclical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclical features using sin/cos transformation.

        Why? Hour 23 and hour 0 are 1 hour apart, but as raw numbers
        they're 23 apart. Sin/cos encoding wraps them correctly:
            hour_sin = sin(2π * hour / 24)
            hour_cos = cos(2π * hour / 24)
        This tells the model that 23:00 and 00:00 are neighbors.
        """
        # Hour cycle (24h)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week cycle (7 days)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Month cycle (12 months)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    # ── Holiday Features ───────────────────────────────────────────────

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add German public holiday flags."""
        df["is_holiday"] = df["timestamp"].dt.date.apply(
            lambda d: int(d in self.german_holidays)
        )
        # Day before/after holiday also has different patterns
        df["is_holiday_eve"] = df["is_holiday"].shift(-24, fill_value=0)
        df["is_post_holiday"] = df["is_holiday"].shift(24, fill_value=0)
        return df

    # ── Lag Features ───────────────────────────────────────────────────

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features — load value N hours ago.

        Key lags:
          1h  — very recent trend
          24h — same hour yesterday (strong daily pattern)
          48h — same hour 2 days ago
          168h — same hour last week (strongest predictor for weekday patterns)
        """
        lag_hours = FEAT_PARAMS["lag_hours"]
        for lag in lag_hours:
            df[f"lag_{lag}h"] = df["actual_load_mw"].shift(lag)
        return df

    # ── Rolling Statistics ─────────────────────────────────────────────

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling mean and std over different windows.

        Rolling mean   — smoothed trend (less noisy than raw lag)
        Rolling std    — volatility signal (high std = unusual period)
        """
        windows = FEAT_PARAMS["rolling_windows"]
        for window in windows:
            df[f"rolling_mean_{window}h"] = (
                df["actual_load_mw"].shift(1).rolling(window=window).mean()
            )
            df[f"rolling_std_{window}h"] = (
                df["actual_load_mw"].shift(1).rolling(window=window).std()
            )
        return df

    # ── Difference Features ────────────────────────────────────────────

    def _add_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rate of change features.
        How much did load change vs 1h ago? vs 24h ago?
        Captures momentum — is consumption rising or falling?
        """
        df["diff_1h"]  = df["actual_load_mw"].diff(1)
        df["diff_24h"] = df["actual_load_mw"].diff(24)
        df["diff_168h"]= df["actual_load_mw"].diff(168)
        return df

    # ── Main Interface ─────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data and transform.
        Call this ONLY on training data.
        """
        logger.info("Fitting feature engineer on training data...")
        df = self._build_features(df)
        df = df.dropna()   # Drop rows with NaN from lags

        # Store which columns are features (not target or metadata)
        self.feature_columns = [
            c for c in df.columns
            if c not in ["timestamp", "actual_load_mw"]
        ]
        self.is_fitted = True
        logger.info(f"Feature engineering complete: {len(self.feature_columns)} features")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted parameters.
        Call this on validation, test, and live prediction data.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        df = self._build_features(df)
        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: apply all feature engineering steps in order."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        df = self._add_calendar_features(df)
        df = self._add_cyclical_encoding(df)
        df = self._add_holiday_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_diff_features(df)

        return df

    def save(self, path: Path = Path("models/feature_engineer.pkl")):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Feature engineer saved to {path}")

    @staticmethod
    def load(path: Path = Path("models/feature_engineer.pkl")) -> "EnergyFeatureEngineer":
        engineer = joblib.load(path)
        logger.info(f"Feature engineer loaded from {path}")
        return engineer


def run_preprocessing():
    """Main entry point — load raw data, engineer features, save."""
    raw_path = Path(DATA_PARAMS["raw_path"])

    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run fetch.py first.")
        raise FileNotFoundError(f"{raw_path} not found")

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    logger.info(f"Raw data shape: {df.shape}")

    # Split before fitting to avoid data leakage
    test_days = DATA_PARAMS["test_split_days"]
    split_point = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["timestamp"] <= split_point].copy()
    test_df  = df[df["timestamp"] >  split_point].copy()

    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # Fit on train, transform both
    engineer = EnergyFeatureEngineer()
    train_features = engineer.fit_transform(train_df)
    test_features  = engineer.transform(test_df)

    # Save processed data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_features = pd.concat([train_features, test_features]).sort_values("timestamp")
    full_features.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Saved processed features to {PROCESSED_PATH}")
    logger.info(f"Final shape: {full_features.shape}")

    # Save fitted engineer
    engineer.save()

    return full_features, engineer


if __name__ == "__main__":
    df, eng = run_preprocessing()
    print(f"\nFeature columns ({len(eng.feature_columns)}):")
    for col in eng.feature_columns:
        print(f"  {col}")
    print(f"\nSample rows:")
    print(df[["timestamp", "actual_load_mw"] + eng.feature_columns[:5]].head())
