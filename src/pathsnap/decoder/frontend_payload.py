"""Convert PathSnap decoder results to a JSON-serialisable frontend payload.

The decoder stores coordinates as [latitude, longitude] for Folium.
GeoJSON requires [longitude, latitude]. Keep route segments separate: joining them
would invent roads across HMM gaps. No labels or reference routes are accessed.
"""
from __future__ import annotations

import json
import math
from typing import Any

from .types import TripDecodeResult


def _json_safe(value: Any) -> Any:
    """Convert common numeric/array scalars and reject non-finite values."""
    if hasattr(value, 'item') and callable(value.item):
        value = value.item()
    if hasattr(value, 'tolist') and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def decoded_route_geojson(result: TripDecodeResult) -> dict[str, Any]:
    """Create a GeoJSON FeatureCollection, one LineString per valid segment.

    Degenerate segments with fewer than two points are not rendered as lines.
    They remain available through ``route_segments`` and ``route_geometry`` in
    the payload. Reject invalid coordinates instead of silently reordering them.
    """
    features = []
    for i, segment in enumerate(result.route_geometry):
        coords = []
        for point in segment:
            if len(point) != 2:
                raise ValueError(f'route segment {i} has an invalid coordinate pair')
            lat, lon = (float(point[0]), float(point[1]))
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f'route segment {i} has an invalid latitude/longitude')
            coords.append([lon, lat])
        if len(coords) < 2:
            continue
        features.append({
            'type': 'Feature',
            'properties': {'trip_id': result.trip_id, 'segment_index': i,
                           'arc_ids': list(result.route_segments[i]) if i < len(result.route_segments) else []},
            'geometry': {'type': 'LineString', 'coordinates': coords},
        })
    return {'type': 'FeatureCollection', 'features': features}


def decoder_frontend_payload(result: TripDecodeResult) -> dict[str, Any]:
    """Return a transport-neutral JSON payload for Member 4's FastAPI handler."""
    payload = {
        'trip_id': result.trip_id,
        'status': result.status,
        'observation_indices': result.observation_indices,
        'matched_arc_ids': result.matched_arc_ids,
        'observation_status': result.observation_status,
        'expanded_route_arc_ids': result.expanded_route_arc_ids,
        'route_segments': result.route_segments,
        'gap_boundaries': result.gap_boundaries,
        'diagnostics': result.diagnostics,
        'runtime_seconds': result.runtime_seconds,
        'route_geojson': decoded_route_geojson(result),
    }
    safe = _json_safe(payload)
    json.dumps(safe, allow_nan=False)  # fail early on any unsupported payload types
    return safe
