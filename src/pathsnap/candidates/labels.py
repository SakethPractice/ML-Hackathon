"""Per-observation reference labels and candidate miss rate.

Only used where the reference route may be read: training trips, and the
student-testing trips for development-stage evaluation. The prediction path
must not import this module.

Label rule
----------
Each observation gets the arc of its aligned route position (see
``pathsnap.data.route_alignment``). The label is marked *valid* only when the
fix lies within ``max_label_dist_m`` of that arc. In the training EDA the
alignment distance is clearly bimodal: fixes on the road sit within a few
metres, and a second group at 100-200 m consists mostly of parked vehicles
off the road network and stretches where the GPS goes where the route does
not. Those fixes have no defensible reference arc, so they are excluded from
label-based metrics and reported, not silently relabelled.

``ref_has_duplicate`` marks reference arcs for which another arc has exactly
the same from/to nodes. Such arcs are geometrically identical, so no
GPS-based method can tell them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.candidates.generator import CandidateSet
from pathsnap.data.loader import Trip
from pathsnap.data.route_alignment import align_track_to_route
from pathsnap.graph.road_graph import RoadGraph

DEFAULT_MAX_LABEL_DIST_M = 50.0


def duplicate_arc_mask(graph: RoadGraph) -> np.ndarray:
    """True for arcs sharing (from_node, to_node) with at least one other arc."""
    key = graph.arc_from * graph.n_nodes + graph.arc_to
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inverse] > 1


def reference_labels(trip: Trip, graph: RoadGraph, track_xy: np.ndarray,
                     max_label_dist_m: float = DEFAULT_MAX_LABEL_DIST_M) -> pd.DataFrame:
    if not trip.has_route:
        raise ValueError(f"trip {trip.trip_id} has no reference route")
    al = align_track_to_route(track_xy, trip.route["arc_id"].to_numpy(), graph)
    al.index = trip.track.index
    al["label_valid"] = al["align_dist_m"] <= max_label_dist_m
    al["ref_has_duplicate"] = duplicate_arc_mask(graph)[al["ref_arc_id"].to_numpy()]
    return al


def candidate_hits(candidates: CandidateSet, labels: pd.DataFrame) -> np.ndarray:
    """Per observation: is the reference arc among its candidates?"""
    return candidates.contains(labels.index.to_numpy(), labels["ref_arc_id"].to_numpy())


def candidate_miss_rate(candidates: CandidateSet, labels: pd.DataFrame) -> dict:
    """Miss rate over observations with a valid label.

    Denominator: observations with ``label_valid``. Observations without any
    candidate count as misses.
    """
    valid = labels["label_valid"].to_numpy()
    hit = candidate_hits(candidates, labels)
    n_valid = int(valid.sum())
    return {
        "n_obs": len(labels),
        "n_label_valid": n_valid,
        "n_label_excluded": int((~valid).sum()),
        "n_miss": int((~hit & valid).sum()),
        "candidate_miss_rate": float((~hit[valid]).mean()) if n_valid else np.nan,
        "n_obs_without_candidates": int(len(candidates.empty_obs)),
        "mean_candidates_per_obs": float(len(candidates.table) / max(1, candidates.n_obs)),
    }
