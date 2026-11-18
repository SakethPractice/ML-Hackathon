"""Translate a single candidate-pipeline trip into inputs for hmm.decode_trip."""

import numpy as np
import pandas as pd

from pathsnap.candidates.pipeline import PreparedTrip
from pathsnap.decoder.network import DirectedRoadNetwork


def decoder_inputs(trip: PreparedTrip, scores: pd.DataFrame):
    """Accept one trip's scores keyed by obs_id and arc_id, in any row order.

    Prepare inference trips with with_labels=False. Scores must cover the
    candidate table exactly. Synthetic scores belong in integration tests.
    """
    observations = trip.trip.track[["lon", "lat", "timestamp"]].rename_axis(
        "observation_idx"
    ).reset_index()
    keys = ["obs_id", "arc_id"]
    if "trip_id" in scores and not scores["trip_id"].eq(trip.trip.trip_id).all():
        raise ValueError("scores contain a different trip_id")
    candidates = trip.candidates.table[[*keys, "t_along"]].merge(
        scores[[*keys, "match_probability"]], on=keys, how="outer",
        validate="one_to_one", indicator=True,
    )
    if not candidates["_merge"].eq("both").all():
        raise ValueError("scores must cover exactly this trip's candidate keys")
    candidates = candidates.drop(columns="_merge").rename(
        columns={"obs_id": "observation_idx", "t_along": "proj_fraction"}
    )
    graph = trip.graph
    network = DirectedRoadNetwork(
        graph.node_lonlat, np.column_stack((graph.arc_from, graph.arc_to)),
        arc_length_m=graph.arc_length_m,
    )
    return observations, candidates, network
