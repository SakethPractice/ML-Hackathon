# PathSnap — decoder → API/frontend contract

## State

This handoff can be implemented **before Member 2 delivers trained probabilities**.
It does not claim real model accuracy or final integration. `decoder_frontend_payload`
is a JSON-formatting helper, **not a FastAPI endpoint**.

## Consumer flow (Member 4)

1. Receive GPS/selected trip from the API request.
2. M1 loads trip, directed road network, and candidate arcs.
3. M2 scores each candidate: `(obs_id, arc_id, match_probability)` with
   the correct trip identifier; score **all** candidates, preserve IDs.
4. M3 `decoder_inputs(prepared_trip, scored_rows)` converts input schema;
   `decode_trip(trip_id, obs, candidates, network)` produces `TripDecodeResult`.
5. Call `decoder_frontend_payload(result)` and return its mapping from FastAPI;
   the frontend draws each feature in `route_geojson.features` separately.

```python
from pathsnap.decoder.hmm import decode_trip
from pathsnap.decoder.pipeline_adapter import decoder_inputs
from pathsnap.decoder.frontend_payload import decoder_frontend_payload

# prepared_trip is created by Member 1's candidate pipeline with with_labels=False
# scored_rows is supplied by Member 2 (or explicitly synthetic in tests)
obs, candidates, network = decoder_inputs(prepared_trip, scored_rows)
result = decode_trip(prepared_trip.trip.trip_id, obs, candidates, network)
response_body = decoder_frontend_payload(result)
# FastAPI route handler can return response_body; do not fabricate model scores.
```

## Response fields

- `trip_id`, `status` (`success`, `success_with_unresolved`, `split`, `failed`)
- `observation_indices`, `matched_arc_ids` (nullable, aligned), `observation_status`
- `expanded_route_arc_ids`, `route_segments` (directed arc IDs)
- `gap_boundaries` (do **not** draw a connecting segment across a gap)
- `route_geojson` (GeoJSON `FeatureCollection`, one `LineString` per drawable segment)
- `diagnostics`, `runtime_seconds`

**Coordinates:** the existing decoder's `route_geometry` uses `[lat, lon]`
(Folium order), while GeoJSON is `[lon, lat]`; this helper transforms them.
For a `split` result, render disconnected route segments separately.

## Before integration merge

- Run `python -m pytest -q tests/test_frontend_payload.py` and complete suite.
- Test API JSON serialization and frontend rendering of success, split,
  unresolved, and failed cases.
- When M2 is ready, ensure probability table has all `(obs_id, arc_id)` keys;
  use validation split for tuning and preserve final test-data isolation.
- Do not run reference-route alignment or label reads on inference requests.

## Suggested frontend demo states

- A successful continuous decoded route.
- A split route with visible disconnected sections and explanatory warning.
- A trip with unresolved observations, showing counts from diagnostics.
- A failed decode, handled gracefully instead of displaying a fake route.
