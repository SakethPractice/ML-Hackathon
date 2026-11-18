"""Pydantic request and response models for the PathSnap HTTP API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TripListResponse(BaseModel):
    """Evaluator trip IDs discovered from the configured raw data partition."""

    trips: list[str]


class PredictRequest(BaseModel):
    """Request a prediction for one evaluator trip."""

    trip_id: str = Field(min_length=1, description="ID from GET /trips")


class PredictionResponse(BaseModel):
    """Serialized ``TripDecodeResult`` returned by the HMM/Viterbi decoder.

    ``route_geometry`` contains continuous route segments, each represented as
    ``[[latitude, longitude], ...]`` points. It mirrors the existing decoder
    result contract instead of introducing a separate prediction format.
    """

    trip_id: str
    status: Literal["success", "success_with_unresolved", "split", "failed"]
    observation_indices: list[int]
    matched_arc_ids: list[int | None]
    observation_status: list[str]
    expanded_route_arc_ids: list[int]
    route_segments: list[list[int]]
    route_geometry: list[list[list[float]]]
    gap_boundaries: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    runtime_seconds: float
