"""Checks on a loaded trip.

These checks only report problems; they do not modify the data. Cleaning
decisions are made in ``preprocessing.py`` so they stay in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.data.loader import Trip

# Lon/lat plausibility; anything outside is not a valid WGS84 coordinate.
LON_RANGE = (-180.0, 180.0)
LAT_RANGE = (-90.0, 90.0)


def _out_of_range(df: pd.DataFrame) -> int:
    bad = ~df["lon"].between(*LON_RANGE) | ~df["lat"].between(*LAT_RANGE)
    return int(bad.sum())


def validate_track(track: pd.DataFrame) -> dict:
    ts = track["timestamp"].to_numpy()
    dt = np.diff(ts)
    lonlat = track[["lon", "lat"]]
    return {
        "n_obs": len(track),
        "n_missing_values": int(track.isna().sum().sum()),
        "n_out_of_range_coords": _out_of_range(track),
        "n_duplicate_rows": int(track.duplicated().sum()),
        "n_duplicate_positions": int(lonlat.duplicated().sum()),
        "n_duplicate_timestamps": int(pd.Series(ts).duplicated().sum()),
        "n_non_increasing_steps": int((dt <= 0).sum()),
        "timestamps_sorted": bool(np.all(dt >= 0)) if len(dt) else True,
        "duration_s": float(ts[-1] - ts[0]) if len(ts) else 0.0,
        "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "share_dt_eq_1s": float(np.mean(np.isclose(dt, 1.0))) if len(dt) else np.nan,
        "n_gaps_gt_5s": int((dt > 5).sum()),
        "n_gaps_gt_30s": int((dt > 30).sum()),
    }


def validate_network(nodes: pd.DataFrame, arcs: pd.DataFrame) -> dict:
    n_nodes = len(nodes)
    endpoints = arcs[["from_node", "to_node"]].to_numpy()
    invalid_ref = (endpoints < 0) | (endpoints >= n_nodes)
    pairs = arcs[["from_node", "to_node"]]
    return {
        "n_nodes": n_nodes,
        "n_arcs": len(arcs),
        "n_missing_node_values": int(nodes.isna().sum().sum()),
        "n_out_of_range_node_coords": _out_of_range(nodes),
        "n_duplicate_node_coords": int(nodes.duplicated().sum()),
        "n_arcs_invalid_node_ref": int(invalid_ref.any(axis=1).sum()),
        "n_self_loops": int((arcs["from_node"] == arcs["to_node"]).sum()),
        "n_duplicate_arcs": int(pairs.duplicated().sum()),
    }


def validate_route(route: pd.DataFrame, arcs: pd.DataFrame) -> dict:
    """Route checks: valid arc IDs and whether consecutive arcs share a node."""
    r = route["arc_id"].to_numpy()
    n_arcs = len(arcs)
    valid = (r >= 0) & (r < n_arcs)
    out = {
        "route_len": len(r),
        "route_n_unique_arcs": int(len(np.unique(r))),
        "route_n_invalid_arc_ids": int((~valid).sum()),
        "route_n_repeated_consecutive": int((r[1:] == r[:-1]).sum()) if len(r) > 1 else 0,
    }
    if len(r) > 1 and valid.all():
        frm = arcs["from_node"].to_numpy()
        to = arcs["to_node"].to_numpy()
        joined = to[r[:-1]] == frm[r[1:]]
        out["route_n_breaks"] = int((~joined).sum())
        out["route_share_connected"] = float(joined.mean())
    else:
        out["route_n_breaks"] = np.nan
        out["route_share_connected"] = np.nan
    return out


def validate_trip(trip: Trip) -> dict:
    report = {"trip_id": trip.trip_id, "partition": trip.partition}
    report.update(validate_track(trip.track))
    report.update(validate_network(trip.nodes, trip.arcs))
    report["has_route"] = trip.has_route
    if trip.has_route:
        report.update(validate_route(trip.route, trip.arcs))
    return report
