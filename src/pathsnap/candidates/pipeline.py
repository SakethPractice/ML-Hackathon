"""Single entry point from raw trip files to candidates.

    from pathsnap.candidates.pipeline import load_data_config, prepare_trip

    cfg = load_data_config()
    pt = prepare_trip("train", "00000001", cfg, with_labels=True)
    pt.candidates.table   # one row per (obs_id, arc_id)
    pt.context            # one row per obs_id, GPS-only context
    pt.labels             # one row per obs_id, reference arc + label_valid

With ``with_labels=False`` the route file is never read; that is the mode the
API / inference path must use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from pathsnap.candidates.generator import CandidateConfig, CandidateGenerator, CandidateSet
from pathsnap.data.loader import Trip, load_trip
from pathsnap.data.preprocessing import TrackContextConfig, track_context
from pathsnap.graph.road_graph import RoadGraph

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")


@dataclass(frozen=True)
class DataConfig:
    data_root: str = "data/raw"
    candidates: CandidateConfig = CandidateConfig()
    context: TrackContextConfig = TrackContextConfig()
    max_label_dist_m: float = 50.0


def load_data_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DataConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cand = CandidateConfig(
        radius_m=raw["candidate_radius_metres"],
        top_k=raw["candidate_top_k"],
        fallback_factor=raw["candidate_fallback_factor"],
        max_radius_m=raw["candidate_max_radius_metres"],
        index_step_m=raw["candidate_index_step_metres"],
    )
    ctx = TrackContextConfig(
        gap_s=raw["gap_seconds"],
        heading_lag=raw["heading_lag"],
        min_heading_disp_m=raw["min_heading_disp_metres"],
        stationary_window=raw["stationary_window"],
        stationary_disp_m=raw["stationary_disp_metres"],
    )
    return DataConfig(raw["data_root"], cand, ctx, raw["max_label_dist_metres"])


@dataclass
class PreparedTrip:
    trip: Trip
    graph: RoadGraph
    context: pd.DataFrame
    candidates: CandidateSet
    labels: pd.DataFrame | None = None


def prepare_trip(partition: str, trip_id: str, cfg: DataConfig | None = None,
                 with_labels: bool = False, data_root: str | Path | None = None) -> PreparedTrip:
    cfg = cfg or DataConfig()
    root = data_root or cfg.data_root
    trip = load_trip(partition, trip_id, root, with_route=with_labels)
    graph = RoadGraph.from_trip(trip)
    context = track_context(trip.track, graph.projection, cfg.context)
    xy = context[["x", "y"]].to_numpy()
    candidates = CandidateGenerator(graph, cfg.candidates).generate(xy)

    labels = None
    if with_labels:
        # imported here so the inference path does not even load the label code
        from pathsnap.candidates.labels import reference_labels
        labels = reference_labels(trip, graph, xy, cfg.max_label_dist_m)
    return PreparedTrip(trip, graph, context, candidates, labels)
