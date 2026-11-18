
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathsnap.candidates.pipeline import prepare_trip
from pathsnap.data import loader
from pathsnap.decoder.hmm import decode_trip
from pathsnap.decoder.pipeline_adapter import decoder_inputs


def test_training_trip_with_synthetic_probabilities(monkeypatch):
    # Locate the raw dataset.
    data_root = Path(
        os.environ.get("PATHSNAP_DATA_ROOT", "data/raw")
    )
    trip_dir = data_root / "train" / "00000039"

    # Raw data is not always committed to GitHub.
    if not trip_dir.is_dir():
        pytest.skip(
            f"Raw training trip not found: {trip_dir}"
        )

    # Ensure the inference pipeline never reads reference labels.
    def forbid_reference(*args, **kwargs):
        pytest.fail("Inference attempted to read a reference route")

    monkeypatch.setattr(loader, "read_route", forbid_reference)

    # Load real GPS observations and candidate road segments.
    trip = prepare_trip(
        "train",
        "00000039",
        with_labels=False,
        data_root=data_root,
    )

    raw = trip.candidates.table

    # Synthetic probabilities for integration testing only.
    # These are NOT trained-model predictions.
    scores = raw[["obs_id", "arc_id"]].assign(
        match_probability=1 / (1 + raw["rank"])
    )

    # Reverse row order to verify key-based matching.
    obs, candidates, network = decoder_inputs(
        trip,
        scores.iloc[::-1],
    )

    # Verify observation mapping.
    pd.testing.assert_frame_equal(
        obs.set_index("observation_idx").rename_axis("obs_id"),
        trip.trip.track[["lon", "lat", "timestamp"]],
    )

    # Verify candidate projection fractions.
    mapped = candidates.set_index(
        ["observation_idx", "arc_id"]
    ).sort_index()

    original = raw.set_index(
        ["obs_id", "arc_id"]
    ).sort_index()

    np.testing.assert_array_equal(
        mapped["proj_fraction"],
        original["t_along"],
    )

    # Verify synthetic scores are assigned to correct candidates.
    expected_scores = scores.set_index(
        ["obs_id", "arc_id"]
    ).sort_index()

    np.testing.assert_array_equal(
        mapped["match_probability"],
        expected_scores["match_probability"],
    )

    # Verify directed graph ID and geometry mapping.
    np.testing.assert_array_equal(
        network.arcs,
        trip.trip.arcs[["from_node", "to_node"]].to_numpy(),
    )

    np.testing.assert_array_equal(
        network.node_lonlat,
        trip.graph.node_lonlat,
    )

    np.testing.assert_allclose(
        network.arc_length,
        trip.graph.arc_length_m,
    )

    # Run the actual HMM/Viterbi decoder.
    result = decode_trip(
        trip.trip.trip_id,
        obs,
        candidates,
        network,
    )

    assert result.observation_indices == trip.trip.track.index.tolist()
    assert result.route_segments

    # Every matched arc must be a candidate for that observation.
    for idx, arc in zip(
        result.observation_indices,
        result.matched_arc_ids,
        strict=True,
    ):
        if arc is not None:
            assert (idx, arc) in mapped.index

    # Verify directed route connectivity.
    for segment in result.route_segments:
        for a, b in zip(segment, segment[1:]):
            assert network.to_node(a) == network.from_node(b)

    # Missing candidate scores must raise an error.
    with pytest.raises(ValueError, match="candidate keys"):
        decoder_inputs(trip, scores.iloc[1:])
