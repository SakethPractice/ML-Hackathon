import json

import pytest

from pathsnap.decoder.frontend_payload import decoder_frontend_payload, decoded_route_geojson
from pathsnap.decoder.types import TripDecodeResult


def _example(**overrides):
    d = dict(
        trip_id='00000039', status='split',
        observation_indices=[0, 1, 2], matched_arc_ids=[4, None, 9],
        observation_status=['decoded', 'no_candidates', 'decoded'],
        expanded_route_arc_ids=[4, 5, 9], route_segments=[[4, 5], [9]],
        route_geometry=[[[12.9, 77.6], [12.91, 77.61]],
                        [[13.0, 77.7], [13.01, 77.71]]],
        gap_boundaries=[{'after_observation_idx': 0, 'before_observation_idx': 2,
                         'reason': 'missing_candidates', 'expanded_route_index': 2}],
        diagnostics={'n_decoded_observations': 2}, runtime_seconds=0.25,
    )
    d.update(overrides)
    return TripDecodeResult(**d)


def test_geojson_preserves_gaps_and_flips_coordinate_order():
    result = _example()
    geo = decoded_route_geojson(result)
    assert geo['type'] == 'FeatureCollection'
    assert len(geo['features']) == 2
    assert geo['features'][0]['geometry']['coordinates'][0] == [77.6, 12.9]
    assert geo['features'][1]['properties']['arc_ids'] == [9]
    assert geo['features'][0]['geometry']['type'] == 'LineString'
    assert geo['features'][0]['properties']['segment_index'] == 0


def test_payload_json_and_unresolved_observations():
    payload = decoder_frontend_payload(_example())
    assert payload['matched_arc_ids'] == [4, None, 9]
    assert payload['status'] == 'split'
    assert payload['gap_boundaries'][0]['reason'] == 'missing_candidates'
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_invalid_coordinates_rejected():
    with pytest.raises(ValueError, match='invalid latitude/longitude'):
        decoded_route_geojson(_example(route_geometry=[[[95, 77], [13, 77]]]))


def test_empty_and_single_point_segments_are_safe():
    geo = decoded_route_geojson(_example(route_segments=[[], [1]],
                                        route_geometry=[[], [[12, 77]]]))
    assert geo['features'] == []
