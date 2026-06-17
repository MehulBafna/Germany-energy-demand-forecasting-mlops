"""
fetch.py — Pull real German electricity consumption data from ENTSO-E.

ENTSO-E is the European Network of Transmission System Operators.
They provide free hourly energy data for all European countries.

Setup:
    1. Register at https://transparency.entsoe.eu
    2. Request API token (Settings → Web API Security Token)
    3. Add to .env: ENTSOE_API_KEY=your_token_here
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import yaml
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

DATA_PARAMS = params["data"]
COUNTRY_CODE = DATA_PARAMS["country_code"]      # "DE" for Germany
RAW_PATH = Path(DATA_PARAMS["raw_path"])


def fetch_entso_data(
    start_date: str,
    end_date: str,
    country_code: str = COUNTRY_CODE,
) -> pd.DataFrame:
    """
    Fetch actual electricity load data from ENTSO-E API.

    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
        country_code: "DE" for Germany

    Returns:
        DataFrame with columns: [timestamp, actual_load_mw]
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        logger.warning("ENTSOE_API_KEY not set — using fallback synthetic data for dev")
        return _generate_synthetic_data(start_date, end_date)

    try:
        from entsoe import EntsoePandasClient
        client = EntsoePandasClient(api_key=api_key)

        start = pd.Timestamp(start_date, tz="Europe/Berlin")
        end   = pd.Timestamp(end_date,   tz="Europe/Berlin")

        logger.info(f"Fetching ENTSO-E data: {country_code} | {start_date} → {end_date}")

        # Actual total load in MW
        load_series = client.query_load(country_code, start=start, end=end)

        # Convert to DataFrame
        df = load_series.reset_index()
        df.columns = ["timestamp", "actual_load_mw"]
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(df):,} hourly records")
        return df

    except Exception as e:
        logger.error(f"ENTSO-E fetch failed: {e}")
        logger.warning("Falling back to synthetic data")
        return _generate_synthetic_data(start_date, end_date)


def _generate_synthetic_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate realistic synthetic German energy data for development.
    Mirrors real patterns: daily peaks, weekly cycles, seasonal variation.
    Used when ENTSO-E API key is not available.
    """
    import numpy as np
    logger.info("Generating synthetic energy data for development...")

    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(timestamps)

    # Base load ~50,000 MW for Germany
    base = 50000

    # Hour of day pattern (peak at 9-11am and 7-9pm, low at 3-5am)
    hour_pattern = np.array([
        -8000, -9000, -9500, -9000, -8000, -5000,  # 0-5am  (night low)
         -2000,  1000,  4000,  5000,  5500,  5000,  # 6-11am (morning rise)
          4500,  4000,  4500,  5000,  4500,  4000,  # 12-5pm (afternoon)
          5000,  5500,  5000,  3000,  1000, -2000,  # 6-11pm (evening peak)
    ])

    # Day of week pattern (weekends ~15% lower)
    dow_pattern = {0: 0, 1: 500, 2: 500, 3: 500, 4: 1000, 5: -5000, 6: -7000}

    # Seasonal pattern (winter higher, summer lower)
    month_pattern = {
        1: 8000, 2: 7000, 3: 3000, 4: -1000, 5: -3000, 6: -4000,
        7: -4500, 8: -4000, 9: -1000, 10: 2000, 11: 5000, 12: 8000
    }

    np.random.seed(42)
    noise = np.random.normal(0, 1500, n)

    load = np.array([
        base
        + hour_pattern[ts.hour]
        + dow_pattern[ts.dayofweek]
        + month_pattern[ts.month]
        + noise[i]
        for i, ts in enumerate(timestamps)
    ])

    # Clip to realistic range
    load = np.clip(load, 25000, 85000)

    df = pd.DataFrame({"timestamp": timestamps, "actual_load_mw": load.astype(int)})
    logger.info(f"Generated {len(df):,} synthetic records | "
                f"Mean: {df['actual_load_mw'].mean():,.0f} MW | "
                f"Range: {df['actual_load_mw'].min():,} – {df['actual_load_mw'].max():,} MW")
    return df


def fetch_recent_data(days_back: int = 30) -> pd.DataFrame:
    """Fetch the most recent N days of data — used for drift monitoring."""
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    return fetch_entso_data(start_date, end_date)


def save_raw_data(df: pd.DataFrame, path: Path = RAW_PATH) -> None:
    """Save raw data to CSV, appending new rows if file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        # Only keep new rows
        last_ts = existing["timestamp"].max()
        new_rows = df[df["timestamp"] > last_ts]
        if len(new_rows) == 0:
            logger.info("No new data to append")
            return
        df = pd.concat([existing, new_rows], ignore_index=True)
        logger.info(f"Appended {len(new_rows):,} new rows")
    else:
        logger.info(f"Creating new raw data file: {path}")

    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df):,} total records to {path}")


def run_fetch():
    """Main entry point — fetch full training data range from params.yaml."""
    df = fetch_entso_data(
        start_date=DATA_PARAMS["start_date"],
        end_date=DATA_PARAMS["end_date"],
    )
    save_raw_data(df)
    return df


if __name__ == "__main__":
    df = run_fetch()
    print(f"\nSample data:")
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Avg load: {df['actual_load_mw'].mean():,.0f} MW")
