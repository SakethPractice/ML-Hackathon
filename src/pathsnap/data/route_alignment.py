"""Align GPS observations with the reference route.

A ``.route`` file lists the arcs travelled, in order, but says nothing about
which GPS observation was on which arc, and the two sequences have different
lengths (many 1 Hz fixes per long arc, several short arcs between two fixes).
To get a per-observation reference arc we solve a monotone alignment:

    choose j_0 <= j_1 <= ... <= j_{N-1}   (positions in the route)
    minimising  sum_i  d(p_i, route[j_i])

where ``d`` is the point-to-segment distance in metres. The ordering
constraint means the vehicle never moves backwards along its route, which
also makes routes that revisit the same arc (loops) work correctly: each
visit is a separate route position. Route positions may be skipped (short
arcs driven between two fixes), and several fixes may share a position.

This is solved exactly with dynamic programming:

    D[i, j] = d(i, j) + min_{k <= j} D[i-1, k]

The inner min is a running prefix minimum, so each row costs O(M).

The result is a *reference* label, not verified truth: GPS error, gaps and
route-file limitations all affect it. Every aligned observation therefore
carries its distance to the assigned arc so that doubtful labels can be
filtered or reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.graph.geo import point_segment_distance
from pathsnap.graph.road_graph import RoadGraph


def align_track_to_route(track_xy: np.ndarray, route_arcs: np.ndarray, graph: RoadGraph) -> pd.DataFrame:
    """Return one row per observation with its aligned route position and arc.

    Columns: ``route_pos``, ``ref_arc_id``, ``align_dist_m``, ``align_t``.
    """
    track_xy = np.asarray(track_xy, dtype=float)
    route_arcs = np.asarray(route_arcs, dtype=np.int64)
    n, m = len(track_xy), len(route_arcs)
    if n == 0 or m == 0:
        return pd.DataFrame(
            {"route_pos": pd.Series(dtype=np.int64), "ref_arc_id": pd.Series(dtype=np.int64),
             "align_dist_m": pd.Series(dtype=float), "align_t": pd.Series(dtype=float)}
        )

    a, b = graph.arc_xy(route_arcs)
    bp_dtype = np.int16 if m < np.iinfo(np.int16).max else np.int32
    # back[i, j] = argmin_{k <= j} D[i-1, k]  (the predecessor position chosen for row i)
    back = np.empty((n, m), dtype=bp_dtype)

    def row_cost(i):
        p = np.broadcast_to(track_xy[i], (m, 2))
        d, _, _ = point_segment_distance(p, a, b)
        return d

    prev = row_cost(0)
    back[0] = np.arange(m)
    idx = np.arange(m)
    for i in range(1, n):
        # running minimum of prev and where it occurs (first occurrence on ties)
        run_min = np.minimum.accumulate(prev)
        is_new_min = np.r_[True, prev[1:] < run_min[:-1]]
        arg = np.maximum.accumulate(np.where(is_new_min, idx, 0))
        back[i] = arg
        prev = row_cost(i) + run_min

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    d, t, _ = point_segment_distance(track_xy, a[pos], b[pos])
    return pd.DataFrame(
        {"route_pos": pos, "ref_arc_id": route_arcs[pos], "align_dist_m": d, "align_t": t}
    )


def route_coverage(alignment: pd.DataFrame, route_len: int) -> dict:
    """How much of the route was matched to at least one observation."""
    used = np.unique(alignment["route_pos"].to_numpy())
    return {
        "route_positions_matched": int(len(used)),
        "route_share_matched": float(len(used) / route_len) if route_len else np.nan,
        "first_route_pos": int(used.min()) if len(used) else -1,
        "last_route_pos": int(used.max()) if len(used) else -1,
    }
