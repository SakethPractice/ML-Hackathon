"""API endpoint and service-boundary tests (no real model or data needed)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pathsnap.api import main, service
from pathsnap.api.service import EvaluatorTripsUnavailable, PredictionService, list_evaluator_trip_ids
from pathsnap.decoder.types import TripDecodeResult


def _result(trip_id="trip-a"):
    return TripDecodeResult(
        trip_id=trip_id, status="success", observation_indices=[0, 1],
        matched_arc_ids=[3, None], observation_status=["matched", "unresolved"],
        expanded_route_arc_ids=[3], route_segments=[[3]],
        route_geometry=[[[49.0, 8.0], [49.001, 8.001]]], gap_boundaries=[],
        diagnostics={}, runtime_seconds=0.1,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "list_evaluator_trip_ids", lambda: ["trip-a"])
    main.app.state.prediction_service = PredictionService(lambda tid: _result(tid))
    yield TestClient(main.app)
    main.app.state.prediction_service = PredictionService(use_default_pipeline=True)


def test_health(client, monkeypatch):
    monkeypatch.setenv("PATHSNAP_MODE", "student")
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["mode"] == "student"
    assert body["partition"] == "student_test"


def test_mode_selects_partition(monkeypatch):
    monkeypatch.setenv("PATHSNAP_MODE", "evaluator")
    assert service._load_data_settings()[1] == "evaluator_test"
    monkeypatch.setenv("PATHSNAP_MODE", "student")
    assert service._load_data_settings()[1] == "student_test"
    monkeypatch.setenv("PATHSNAP_MODE", "nonsense")
    with pytest.raises(EvaluatorTripsUnavailable):
        service._load_data_settings()


def test_trip_discovery_needs_inputs_not_route(tmp_path, monkeypatch):
    trip_dir = tmp_path / "evaluator_test" / "t1"
    trip_dir.mkdir(parents=True)
    for suffix in (".track", ".nodes", ".arcs"):
        (trip_dir / f"t1{suffix}").touch()
    (tmp_path / "evaluator_test" / "incomplete").mkdir()
    monkeypatch.setattr(service, "_load_data_settings", lambda: (Path(tmp_path), "evaluator_test"))
    assert list_evaluator_trip_ids() == ["t1"]


def test_trips(client):
    body = client.get("/trips").json()
    assert body["trips"] == ["trip-a"]


def test_trips_503_when_partition_missing(client, monkeypatch):
    def missing():
        raise EvaluatorTripsUnavailable("partition is missing")

    monkeypatch.setattr(main, "list_evaluator_trip_ids", missing)
    r = client.get("/trips")
    assert r.status_code == 503 and r.json()["detail"] == "partition is missing"


def test_predict_returns_decoder_result(client):
    r = client.post("/predict", json={"trip_id": "trip-a"})
    assert r.status_code == 200
    body = r.json()
    assert body["trip_id"] == "trip-a" and body["route_geometry"] == [[[49.0, 8.0], [49.001, 8.001]]]
    assert body["matched_arc_ids"] == [3, None]
    assert body["evaluation"] is None  # mocked predictor: no server-side evaluation


def test_predict_unknown_trip_404(client):
    assert client.post("/predict", json={"trip_id": "other"}).status_code == 404


def test_predict_empty_trip_id_422(client):
    assert client.post("/predict", json={"trip_id": ""}).status_code == 422


def test_predict_503_without_pipeline(client):
    main.app.state.prediction_service = PredictionService()
    r = client.post("/predict", json={"trip_id": "trip-a"})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


def test_predict_503_when_model_file_missing(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", str(tmp_path / "missing.joblib"))
    main.app.state.prediction_service = PredictionService(use_default_pipeline=True)
    r = client.post("/predict", json={"trip_id": "trip-a"})
    assert r.status_code == 503 and "train_final_model.py" in r.json()["detail"]


def test_prediction_response_has_no_reference_fields(client):
    body = client.post("/predict", json={"trip_id": "trip-a"}).json()
    assert not {"ref_arc_id", "route", "reference_route", "labels"} & set(body)
