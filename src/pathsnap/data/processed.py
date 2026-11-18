"""Processed per-trip tables written to ``data/processed``.

Layout::

    data/processed/<partition>/<trip_id>/
        gps_observations.csv.gz    one row per GPS fix (GPS-only, no labels)
        candidate_segments.csv.gz  one row per (fix, candidate arc)  (no labels)
        labels.csv.gz              one row per fix: reference arc     (train / student_test only)
        candidate_labels.csv.gz    one row per (fix, candidate arc): is_reference

Labels are kept in their own files so the two input tables can be handed to
feature extraction and inference without any reference information in them.

Column names in the two input tables follow the interface of
``pathsnap.features.extract.extract_features``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates.pipeline import PreparedTrip

DEFAULT_PROCESSED_ROOT = Path("data/processed")

GPS_FILE = "gps_observations.csv.gz"
CANDIDATE_FILE = "candidate_segments.csv.gz"
LABEL_FILE = "labels.csv.gz"
CANDIDATE_LABEL_FILE = "candidate_labels.csv.gz"

ID_DTYPES = {"trip_id": str}


def gps_table(pt: PreparedTrip) -> pd.DataFrame:
    tr = pt.trip.track
    ctx = pt.context
    return pd.DataFrame({
        "trip_id": pt.trip.trip_id,
        "observation_id": tr.index.to_numpy(),
        "longitude": tr["lon"].to_numpy(),
        "latitude": tr["lat"].to_numpy(),
        "timestamp": tr["timestamp"].to_numpy(),
        "x_m": ctx["x"].to_numpy(),
        "y_m": ctx["y"].to_numpy(),
        "dt_prev_s": ctx["dt_prev_s"].to_numpy(),
        "step_prev_m": ctx["step_prev_m"].to_numpy(),
        "speed_prev_mps": ctx["speed_prev_mps"].to_numpy(),
        "heading_deg": ctx["heading_deg"].to_numpy(),
        "window_disp_m": ctx["window_disp_m"].to_numpy(),
        "is_stationary": ctx["is_stationary"].to_numpy(),
        "gap_before": ctx["gap_before"].to_numpy(),
        "segment_id": ctx["segment_id"].to_numpy(),
    })


def candidate_table(pt: PreparedTrip) -> pd.DataFrame:
    c = pt.candidates.table
    g = pt.graph
    arcs = c["arc_id"].to_numpy()
    closest = g.projection.to_lonlat(c[["proj_x", "proj_y"]].to_numpy())
    return pd.DataFrame({
        "trip_id": pt.trip.trip_id,
        "observation_id": c["obs_id"].to_numpy(),
        "arc_id": arcs,
        "from_node_id": g.arc_from[arcs],
        "to_node_id": g.arc_to[arcs],
        "distance_to_segment_m": c["distance_m"].to_numpy(),
        "segment_length_m": g.arc_length_m[arcs],
        "segment_bearing_deg": g.arc_bearing_deg[arcs],
        "distance_rank": c["rank"].to_numpy(),
        "t_along": c["t_along"].to_numpy(),
        "closest_longitude": closest[:, 0],
        "closest_latitude": closest[:, 1],
        "radius_used_m": c["radius_used_m"].to_numpy(),
    })


def label_tables(pt: PreparedTrip) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pt.labels is None:
        raise ValueError("prepare the trip with with_labels=True to export labels")
    lab = pt.labels
    labels = pd.DataFrame({
        "trip_id": pt.trip.trip_id,
        "observation_id": lab.index.to_numpy(),
        "route_pos": lab["route_pos"].to_numpy(),
        "ref_arc_id": lab["ref_arc_id"].to_numpy(),
        "align_dist_m": lab["align_dist_m"].to_numpy(),
        "label_valid": lab["label_valid"].to_numpy(),
        "ref_has_duplicate": lab["ref_has_duplicate"].to_numpy(),
    })
    c = pt.candidates.table
    obs = c["obs_id"].to_numpy()
    cand_labels = pd.DataFrame({
        "trip_id": pt.trip.trip_id,
        "observation_id": obs,
        "arc_id": c["arc_id"].to_numpy(),
        "is_reference": c["arc_id"].to_numpy() == lab["ref_arc_id"].to_numpy()[obs],
        "label_valid": lab["label_valid"].to_numpy()[obs],
    })
    return labels, cand_labels


def trip_folder(partition: str, trip_id: str, root: str | Path = DEFAULT_PROCESSED_ROOT) -> Path:
    return Path(root) / partition / trip_id


def save_processed(pt: PreparedTrip, root: str | Path = DEFAULT_PROCESSED_ROOT) -> dict:
    """Write the tables for one trip and return a manifest row."""
    out = trip_folder(pt.trip.partition, pt.trip.trip_id, root)
    out.mkdir(parents=True, exist_ok=True)
    gps = gps_table(pt)
    cand = candidate_table(pt)
    gps.to_csv(out / GPS_FILE, index=False)
    cand.to_csv(out / CANDIDATE_FILE, index=False)
    row = {
        "partition": pt.trip.partition, "trip_id": pt.trip.trip_id,
        "n_observations": len(gps), "n_candidates": len(cand),
        "n_obs_without_candidates": int(len(pt.candidates.empty_obs)),
        "has_labels": pt.labels is not None,
    }
    if pt.labels is not None:
        labels, cand_labels = label_tables(pt)
        labels.to_csv(out / LABEL_FILE, index=False)
        cand_labels.to_csv(out / CANDIDATE_LABEL_FILE, index=False)
        row["n_label_valid"] = int(labels["label_valid"].sum())
        row["n_candidate_rows_reference"] = int(cand_labels["is_reference"].sum())
    else:
        # stale label files from an earlier run must not survive
        for f in (LABEL_FILE, CANDIDATE_LABEL_FILE):
            (out / f).unlink(missing_ok=True)
    return row


def load_processed(partition: str, trip_id: str, root: str | Path = DEFAULT_PROCESSED_ROOT,
                   with_labels: bool = False) -> dict[str, pd.DataFrame]:
    """Read one trip. Label files are only opened when ``with_labels=True``."""
    d = trip_folder(partition, trip_id, root)
    out = {
        "gps_observations": pd.read_csv(d / GPS_FILE, dtype=ID_DTYPES),
        "candidate_segments": pd.read_csv(d / CANDIDATE_FILE, dtype=ID_DTYPES),
    }
    if with_labels:
        out["labels"] = pd.read_csv(d / LABEL_FILE, dtype=ID_DTYPES)
        out["candidate_labels"] = pd.read_csv(d / CANDIDATE_LABEL_FILE, dtype=ID_DTYPES)
    return out


def load_partition(partition: str, root: str | Path = DEFAULT_PROCESSED_ROOT,
                   with_labels: bool = False) -> dict[str, pd.DataFrame]:
    """Concatenate every trip of a partition; ``trip_id`` identifies the group."""
    base = Path(root) / partition
    if not base.is_dir():
        raise FileNotFoundError(f"no processed data for partition {partition!r} under {root}")
    parts: dict[str, list[pd.DataFrame]] = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        for k, df in load_processed(partition, d.name, root, with_labels).items():
            parts.setdefault(k, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def to_float32(df: pd.DataFrame, exclude=("longitude", "latitude", "closest_longitude",
                                          "closest_latitude", "timestamp")) -> pd.DataFrame:
    """Optional memory saver for large concatenated tables."""
    cols = [c for c in df.select_dtypes(np.float64).columns if c not in exclude]
    return df.astype({c: np.float32 for c in cols})
