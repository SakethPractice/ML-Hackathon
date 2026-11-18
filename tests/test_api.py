"""Focused API endpoint and service-boundary tests."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from pathsnap.api import main
from pathsnap.api.service import (
    EvaluatorTripsUnavailable,
    PredictionService,
    list_evaluator_trip_ids,
)
from pathsnap.decoder.types import TripDecodeResult


def dump_model(model):
    """Serialize Pydantic models across supported Pydantic major versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_health_remains_available():
    assert any(route.path == "/health" for route in main.app.routes)
    health_call = main.health()
    with pytest.raises(StopIteration) as response:
        health_call.send(None)
    assert response.value.value == {"status": "ok"}


def test_trip_discovery_uses_configured_raw_inputs_without_route_file(tmp_path, monkeypatch):
    data_root = tmp_path / "raw-data"
    trip_dir = data_root / "evaluator_test" / "trip-from-fixture"
    trip_dir.mkdir(parents=True)
    for suffix in (".track", ".nodes", ".arcs"):
        (trip_dir / f"{trip_dir.name}{suffix}").touch()
    monkeypatch.setattr(
        "pathsnap.api.service._load_data_settings",
        lambda: (Path(data_root), "evaluator_test"),
    )

    assert list_evaluator_trip_ids() == ["trip-from-fixture"]


def test_trips_endpoint_returns_source_ids(monkeypatch):
    monkeypatch.setattr(main, "list_evaluator_trip_ids", lambda: ["trip-from-source"])

    response = main.get_trips()

    assert dump_model(response) == {"trips": ["trip-from-source"]}


def test_trips_endpoint_reports_missing_evaluator_source(monkeypatch):
    def missing_source():
        raise EvaluatorTripsUnavailable("evaluator source is missing")

    monkeypatch.setattr(main, "list_evaluator_trip_ids", missing_source)

    with pytest.raises(HTTPException) as error:
        main.get_trips()

    assert error.value.status_code == 503
    assert error.value.detail == "evaluator source is missing"


def test_predict_requires_trip_id():
    with pytest.raises(ValidationError):
        main.PredictRequest(**{})


def test_predict_delegates_and_serializes_decoder_result(monkeypatch):
    monkeypatch.setattr(main, "list_evaluator_trip_ids", lambda: ["trip-from-source"])
    result = TripDecodeResult(
        trip_id="trip-from-source",
        status="failed",
        observation_indices=[],
        matched_arc_ids=[],
        observation_status=[],
        expanded_route_arc_ids=[],
        route_segments=[],
        route_geometry=[],
        gap_boundaries=[],
        diagnostics={},
        runtime_seconds=0.0,
    )
    called_with = []

    def mocked_predictor(trip_id):
        called_with.append(trip_id)
        return result

    response = main.predict(
        main.PredictRequest(trip_id="trip-from-source"),
        PredictionService(mocked_predictor),
    )

    assert dump_model(response) == result.to_dict()
    assert called_with == ["trip-from-source"]


def test_predict_returns_404_for_id_outside_evaluator_source(monkeypatch):
    monkeypatch.setattr(main, "list_evaluator_trip_ids", lambda: ["available-trip"])

    with pytest.raises(HTTPException) as error:
        main.predict(
            main.PredictRequest(trip_id="other-trip"),
            PredictionService(),
        )

    assert error.value.status_code == 404


def test_predict_reports_unconfigured_inference_pipeline(monkeypatch):
    monkeypatch.setattr(main, "list_evaluator_trip_ids", lambda: ["trip-from-source"])

    with pytest.raises(HTTPException) as error:
        main.predict(
            main.PredictRequest(trip_id="trip-from-source"),
            PredictionService(),
        )

    assert error.value.status_code == 503
    assert "pipeline is not configured" in error.value.detail
