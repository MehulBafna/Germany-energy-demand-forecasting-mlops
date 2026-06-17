import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.models.train import compute_metrics, time_series_cv, _maybe_promote_to_production


def make_features_df(n: int = 400) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="h")
    load = 50_000 + 5_000 * np.sin(np.arange(n) * 2 * np.pi / 24) + np.random.normal(0, 500, n)
    return pd.DataFrame({
        "timestamp":       timestamps,
        "actual_load_mw":  load,
        "hour":            [t.hour for t in timestamps],
        "day_of_week":     [t.dayofweek for t in timestamps],
        "month":           [t.month for t in timestamps],
        "lag_1h":          np.roll(load, 1),
        "lag_24h":         np.roll(load, 24),
        "rolling_mean_24h": pd.Series(load).rolling(24, min_periods=1).mean().values,
    })


class TestComputeMetrics:

    def test_perfect_predictions_give_zero_error(self):
        y = np.array([1_000.0, 2_000.0, 3_000.0])
        m = compute_metrics(y, y)
        assert m["mae"]  == 0.0
        assert m["rmse"] == 0.0
        assert m["mape"] == 0.0

    def test_known_mae(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 310.0])
        assert compute_metrics(y_true, y_pred)["mae"] == pytest.approx(10.0)

    def test_known_mape(self):
        # 10% error on 100 and 10% error on 200 → MAPE = 10%
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        assert compute_metrics(y_true, y_pred)["mape"] == pytest.approx(10.0)

    def test_returns_mae_rmse_mape_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        assert set(compute_metrics(y, y + 1).keys()) == {"mae", "rmse", "mape"}

    def test_zero_actuals_do_not_cause_division_error(self):
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 210.0])
        m = compute_metrics(y_true, y_pred)
        assert np.isfinite(m["mape"])

    def test_rmse_is_at_least_as_large_as_mae(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 180.0, 350.0, 390.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] >= m["mae"]

    def test_asymmetric_errors_reported_correctly(self):
        # One large error should inflate RMSE more than MAE
        y_true = np.array([100.0, 100.0, 100.0, 100.0])
        y_pred = np.array([100.0, 100.0, 100.0, 200.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] > m["mae"]


class TestTimeSeriesCV:

    def test_returns_correct_keys(self):
        df = make_features_df(300)
        feat_cols = [c for c in df.columns if c not in ["timestamp", "actual_load_mw"]]
        result = time_series_cv(df, feat_cols, "actual_load_mw")
        assert set(result.keys()) == {"mae", "rmse", "mape"}

    def test_all_metrics_are_non_negative(self):
        df = make_features_df(300)
        feat_cols = [c for c in df.columns if c not in ["timestamp", "actual_load_mw"]]
        result = time_series_cv(df, feat_cols, "actual_load_mw")
        assert result["mae"]  >= 0
        assert result["rmse"] >= 0
        assert result["mape"] >= 0

    def test_cv_mae_is_finite(self):
        df = make_features_df(300)
        feat_cols = [c for c in df.columns if c not in ["timestamp", "actual_load_mw"]]
        result = time_series_cv(df, feat_cols, "actual_load_mw")
        assert np.isfinite(result["mae"])


class TestChampionChallenger:

    def _mock_client(self, champion_mae=None):
        client = MagicMock()
        if champion_mae is None:
            client.get_model_version_by_alias.side_effect = Exception("No alias found")
        else:
            champion_mv = MagicMock()
            champion_mv.run_id = "old_run_id"
            champion_mv.version = "1"
            client.get_model_version_by_alias.return_value = champion_mv
            old_run = MagicMock()
            old_run.data.metrics = {"test_mae": champion_mae}
            client.get_run.return_value = old_run
        return client

    def test_promotes_first_model_automatically(self):
        mock_mv = MagicMock(version="1")
        with patch("src.models.train.mlflow.tracking.MlflowClient", return_value=self._mock_client()), \
             patch("src.models.train._register_model", return_value=mock_mv) as mock_reg:
            result = _maybe_promote_to_production("run_abc", new_mae=1000.0)

        assert result is True
        mock_reg.assert_called_once()

    def test_promotes_challenger_when_improvement_exceeds_threshold(self):
        # champion MAE=1000, new MAE=950 → 5% improvement > 2% threshold
        mock_mv = MagicMock(version="2")
        with patch("src.models.train.mlflow.tracking.MlflowClient", return_value=self._mock_client(champion_mae=1000.0)), \
             patch("src.models.train._register_model", return_value=mock_mv):
            result = _maybe_promote_to_production("run_new", new_mae=950.0)

        assert result is True

    def test_does_not_promote_when_below_threshold(self):
        # champion MAE=1000, new MAE=990 → 1% improvement < 2% threshold
        mock_mv = MagicMock(version="2")
        with patch("src.models.train.mlflow.tracking.MlflowClient", return_value=self._mock_client(champion_mae=1000.0)), \
             patch("src.models.train._register_model", return_value=mock_mv):
            result = _maybe_promote_to_production("run_new", new_mae=990.0)

        assert result is False

    def test_returns_false_on_registry_error(self):
        with patch("src.models.train.mlflow.tracking.MlflowClient", side_effect=Exception("registry down")):
            result = _maybe_promote_to_production("run_abc", new_mae=1000.0)

        assert result is False
