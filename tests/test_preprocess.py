import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.preprocess import EnergyFeatureEngineer


def make_raw_df(n_hours: int = 500) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    np.random.seed(0)
    load = np.random.randint(40_000, 60_000, n_hours).astype(float)
    return pd.DataFrame({"timestamp": timestamps, "actual_load_mw": load})


class TestEnergyFeatureEngineer:

    def test_fit_transform_returns_dataframe(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df())
        assert isinstance(result, pd.DataFrame)

    def test_no_nulls_after_fit_transform(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df(500))
        assert result.isnull().sum().sum() == 0

    def test_calendar_features_present(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df())
        for col in ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"]:
            assert col in result.columns, f"Missing: {col}"

    def test_lag_features_present(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df(300))
        for lag in [1, 2, 3, 6, 12, 24]:
            assert f"lag_{lag}h" in result.columns

    def test_rolling_features_present(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df(300))
        for w in [6, 12, 24]:
            assert f"rolling_mean_{w}h" in result.columns
            assert f"rolling_std_{w}h" in result.columns

    def test_feature_column_count_matches_stored(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df(400))
        derived = [c for c in result.columns if c not in ["timestamp", "actual_load_mw"]]
        assert len(derived) == len(eng.feature_columns)

    def test_transform_raises_if_not_fitted(self):
        eng = EnergyFeatureEngineer()
        with pytest.raises(RuntimeError, match="fit_transform"):
            eng.transform(make_raw_df(100))

    def test_transform_produces_same_columns_as_fit_transform(self):
        df = make_raw_df(400)
        eng = EnergyFeatureEngineer()
        train_result = eng.fit_transform(df.iloc[:300])
        test_result = eng.transform(df.iloc[300:])
        assert set(train_result.columns) == set(test_result.columns)

    def test_lag_1h_uses_past_not_future(self):
        df = make_raw_df(200)
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(df).reset_index(drop=True)
        # At any row, lag_1h must equal actual_load_mw from 1h earlier
        for i in range(1, 5):
            ts = result.loc[i, "timestamp"]
            expected = df.loc[df["timestamp"] == ts - pd.Timedelta(hours=1), "actual_load_mw"]
            if not expected.empty:
                assert result.loc[i, "lag_1h"] == pytest.approx(expected.values[0])

    def test_hour_sin_cos_in_valid_range(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df(200))
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_is_weekend_is_binary(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df(300))
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_is_holiday_is_binary(self):
        eng = EnergyFeatureEngineer()
        result = eng.fit_transform(make_raw_df(300))
        assert set(result["is_holiday"].unique()).issubset({0, 1})

    def test_save_load_roundtrip(self, tmp_path):
        df = make_raw_df(400)
        eng = EnergyFeatureEngineer()
        eng.fit_transform(df)
        save_path = tmp_path / "fe.pkl"
        eng.save(save_path)

        loaded = EnergyFeatureEngineer.load(save_path)
        assert loaded.is_fitted
        assert loaded.feature_columns == eng.feature_columns

    def test_loaded_engineer_can_transform(self, tmp_path):
        df = make_raw_df(400)
        eng = EnergyFeatureEngineer()
        eng.fit_transform(df.iloc[:300])
        eng.save(tmp_path / "fe.pkl")

        loaded = EnergyFeatureEngineer.load(tmp_path / "fe.pkl")
        result = loaded.transform(df.iloc[300:])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
