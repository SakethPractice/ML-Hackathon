"""Pydantic request and response models for the PathSnap HTTP API."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    mode: str
    partition: Optional[str] = None
    model: str
    model_loaded: bool


class TripListResponse(BaseModel):
    """Trip IDs discovered in the partition served by the current mode."""

    mode: str = "evaluator"
    trips: list[str]


class PredictRequest(BaseModel):
    """Request a prediction for one trip."""

    trip_id: str = Field(min_length=1, description="ID from GET /trips")


class Evaluation(BaseModel):
    """Aggregate scores computed on the server. No reference arcs."""

    partition: str
    labelled_observations: int
    pointwise_accuracy: Optional[float] = None
    route_arc_recall: Optional[float] = None
    route_arc_precision: Optional[float] = None


class PredictionResponse(BaseModel):
    """Serialized ``TripDecodeResult`` plus map and timing information.

    ``route_geometry`` holds continuous route segments, each a list of
    ``[latitude, longitude]`` points. ``observations`` holds the input GPS
    fixes as ``[latitude, longitude]``.
    """

    trip_id: str
    status: Literal["success", "success_with_unresolved", "split", "failed"]
    observation_indices: list[int]
    matched_arc_ids: list[Optional[int]]
    observation_status: list[str]
    expanded_route_arc_ids: list[int]
    route_segments: list[list[int]]
    route_geometry: list[list[list[float]]]
    gap_boundaries: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    runtime_seconds: float
    observations: list[list[float]] = []
    timings: dict[str, float] = {}
    evaluation: Optional[Evaluation] = None
