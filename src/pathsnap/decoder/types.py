"""Shared typed structures for the Member 3 decoder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def normalise_trip_id(trip_id: Any) -> str:
    """Canonical trip id: string, zero-padded to 8 digits when purely numeric."""
    s = str(trip_id).strip()
    return s.zfill(8) if s.isdigit() else s


@dataclass(frozen=True)
class ObservationStates:
    """All candidate states (directed arcs) of one GPS observation.

    Candidates are ordered by ascending ``arc_id`` so tie-breaking is deterministic.
    ``fractions`` is the projected position along the arc (0 = from-node, 1 = to-node).
    """

    observation_idx: int
    lon: float
    lat: float
    timestamp: float
    arc_ids: np.ndarray
    fractions: np.ndarray
    distances_m: np.ndarray
    match_scores: np.ndarray
    log_emissions: np.ndarray

    def __len__(self) -> int:
        return int(self.arc_ids.size)

    def select(self, k: int) -> "ObservationStates":
        """Single-candidate view (used for continuity checks of a chosen candidate)."""
        sl = slice(k, k + 1)
        return ObservationStates(self.observation_idx, self.lon, self.lat, self.timestamp,
                                 self.arc_ids[sl], self.fractions[sl], self.distances_m[sl],
                                 self.match_scores[sl], self.log_emissions[sl])


@dataclass(frozen=True)
class DecodedState:
    observation_idx: int
    arc_id: int
    fraction: float
    timestamp: Optional[float] = None


@dataclass
class TripDecodeResult:
    """Output contract of :func:`pathsnap.decoder.hmm.decode_trip`.

    status: ``success`` | ``success_with_unresolved`` | ``split`` | ``failed``
    matched_arc_ids / observation_status are aligned with ``observation_indices``
    (None arc = unresolved observation; see observation_status for the reason).
    route_geometry is a list of polylines (one per continuous route segment),
    each a list of ``[lat, lon]`` points (Folium order).
    """

    trip_id: str
    status: str
    observation_indices: list[int]
    matched_arc_ids: list[Optional[int]]
    observation_status: list[str]
    expanded_route_arc_ids: list[int]
    route_segments: list[list[int]]
    route_geometry: list[list[list[float]]]
    gap_boundaries: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "status": self.status,
            "observation_indices": self.observation_indices,
            "matched_arc_ids": self.matched_arc_ids,
            "observation_status": self.observation_status,
            "expanded_route_arc_ids": self.expanded_route_arc_ids,
            "route_segments": self.route_segments,
            "route_geometry": self.route_geometry,
            "gap_boundaries": self.gap_boundaries,
            "diagnostics": self.diagnostics,
            "runtime_seconds": self.runtime_seconds,
        }
