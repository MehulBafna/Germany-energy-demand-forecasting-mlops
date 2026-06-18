import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    with patch("src.api.app.ModelStore.load"):
        from src.api.app import app as _app
        return _app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def model_ready():
    mock_model = MagicMock()
    mock_model.predict.return_value = [52_000.0] * 48
    with patch("src.api.app.ModelStore.model", mock_model), \
         patch("src.api.app.ModelStore.feature_engineer", MagicMock(is_fitted=True)), \
         patch("src.api.app.ModelStore.feature_columns", ["lag_1h", "hour"]), \
         patch("src.api.app.ModelStore.loaded_at", "2024-01-01T00:00:00"), \
         patch("src.api.app.ModelStore.is_ready", return_value=True):
        yield


class TestHealthEndpoints:

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["service"] == "EnergyPulse API"

    def test_root_shows_docs_link(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_health_degraded_when_model_not_loaded(self, client):
        with patch("src.api.app.ModelStore.is_ready", return_value=False):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_health_healthy_when_model_loaded(self, client):
        with patch("src.api.app.ModelStore.is_ready", return_value=True), \
             patch("src.api.app.ModelStore.loaded_at", "2024-01-01T00:00:00"):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        assert resp.json()["model_loaded"] is True

    def test_health_includes_uptime(self, client):
        resp = client.get("/health")
        assert "uptime_seconds" in resp.json()
        assert resp.json()["uptime_seconds"] >= 0


class TestPredictEndpoint:

    def test_returns_503_when_model_not_loaded(self, client):
        with patch("src.api.app.ModelStore.is_ready", return_value=False):
            resp = client.post("/predict", json={
                "start_timestamp": "2024-06-15 14:00:00",
                "hours_ahead": 5
            })
        assert resp.status_code == 503

    def test_rejects_invalid_timestamp_format(self, client):
        with patch("src.api.app.ModelStore.is_ready", return_value=True):
            resp = client.post("/predict", json={
                "start_timestamp": "15/06/2024 14:00",
                "hours_ahead": 5
            })
        assert resp.status_code == 400

    def test_rejects_hours_above_max(self, client):
        with patch("src.api.app.ModelStore.is_ready", return_value=True):
            resp = client.post("/predict", json={
                "start_timestamp": "2024-06-15 14:00:00",
                "hours_ahead": 99
            })
        assert resp.status_code == 422

    def test_rejects_zero_hours(self, client):
        with patch("src.api.app.ModelStore.is_ready", return_value=True):
            resp = client.post("/predict", json={
                "start_timestamp": "2024-06-15 14:00:00",
                "hours_ahead": 0
            })
        assert resp.status_code == 422

    def test_predict_now_returns_503_when_no_model(self, client):
        with patch("src.api.app.ModelStore.is_ready", return_value=False):
            resp = client.get("/predict/now?hours_ahead=6")
        assert resp.status_code == 503

    def test_predict_response_shape(self, client, model_ready):
        with patch("src.api.app.build_prediction_features") as mock_feat:
            import pandas as pd, numpy as np
            ts = pd.date_range("2024-06-15 14:00:00", periods=3, freq="h")
            mock_feat.return_value = pd.DataFrame({
                "timestamp": ts,
                "lag_1h": [50000.0] * 3,
                "hour": [14, 15, 16],
            })
            resp = client.post("/predict", json={
                "start_timestamp": "2024-06-15 14:00:00",
                "hours_ahead": 3
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "forecast_start" in data
        assert "model_info" in data
        assert len(data["predictions"]) == 3

    def test_prediction_has_bounds(self, client, model_ready):
        with patch("src.api.app.build_prediction_features") as mock_feat:
            import pandas as pd
            ts = pd.date_range("2024-06-15 14:00:00", periods=2, freq="h")
            mock_feat.return_value = pd.DataFrame({
                "timestamp": ts,
                "lag_1h": [50000.0, 50000.0],
                "hour": [14, 15],
            })
            resp = client.post("/predict", json={
                "start_timestamp": "2024-06-15 14:00:00",
                "hours_ahead": 2
            })
        for pred in resp.json()["predictions"]:
            assert pred["lower_bound_mw"] < pred["predicted_load_mw"]
            assert pred["upper_bound_mw"] > pred["predicted_load_mw"]


class TestMonitoringEndpoints:

    def test_metrics_returns_404_when_no_report(self, client, tmp_path):
        report = Path("monitoring/evaluation_report.json")
        original = report.rename(report.with_suffix(".json.bak")) if report.exists() else None
        try:
            resp = client.get("/metrics")
            assert resp.status_code == 404
        finally:
            if original:
                original.rename(report)

    def test_metrics_returns_200_when_report_exists(self, client, tmp_path, monkeypatch):
        report = {"mae_mw": 1200.0, "mape_pct": 2.5, "rmse_mw": 1700.0}
        report_file = tmp_path / "evaluation_report.json"
        report_file.write_text(json.dumps(report))
        monkeypatch.setattr("src.api.app.Path", lambda p: report_file if "evaluation_report" in str(p) else Path(p))
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_drift_returns_404_when_no_report(self, client):
        resp = client.get("/drift")
        assert resp.status_code == 404

    def test_feedback_accepted(self, client):
        resp = client.post("/feedback", json={
            "timestamp": "2024-06-15 14:00:00",
            "predicted_load_mw": 52_000.0,
            "actual_load_mw": 51_500.0,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "feedback logged"

    def test_feedback_with_notes(self, client):
        resp = client.post("/feedback", json={
            "timestamp": "2024-06-15 15:00:00",
            "predicted_load_mw": 53_000.0,
            "actual_load_mw": 52_000.0,
            "notes": "holiday effect",
        })
        assert resp.status_code == 200


class TestAdminEndpoints:

    def test_reload_model_returns_500_when_files_missing(self, client):
        with patch("src.api.app.ModelStore.load"), \
             patch("src.api.app.ModelStore.is_ready", return_value=False):
            resp = client.post("/reload-model")
        assert resp.status_code == 500

    def test_reload_model_returns_200_when_successful(self, client):
        with patch("src.api.app.ModelStore.load"), \
             patch("src.api.app.ModelStore.is_ready", return_value=True), \
             patch("src.api.app.ModelStore.loaded_at", "2024-01-01T12:00:00"):
            resp = client.post("/reload-model")
        assert resp.status_code == 200
        assert "loaded_at" in resp.json()
