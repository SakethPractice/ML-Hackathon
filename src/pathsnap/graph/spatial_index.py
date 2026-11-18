"""Spatial index over road arcs.

Arcs are straight segments between two nodes, and some are several kilometres
long, so indexing only their endpoints or midpoints would miss a GPS point
lying next to the middle of a long arc. Instead every arc is sampled at a
fixed spacing ``step_m`` and the samples go into a KD-tree.

Any point of a segment is within ``step_m / 2`` of one of its samples, so a
ball query of radius ``r + step_m / 2`` returns a superset of all arcs whose
true point-to-segment distance is <= ``r``. The exact distance is then
computed for those arcs and the rest are dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pathsnap.graph.geo import point_segment_distance
from pathsnap.graph.road_graph import RoadGraph


class ArcIndex:
    def __init__(self, graph: RoadGraph, step_m: float = 25.0):
        self.graph = graph
        self.step_m = float(step_m)
        a, b = graph.arc_xy(np.arange(graph.n_arcs))
        seg_len = np.linalg.norm(b - a, axis=1)
        n_samples = np.maximum(1, np.ceil(seg_len / self.step_m).astype(np.int64)) + 1
        owner = np.repeat(np.arange(graph.n_arcs), n_samples)
        # position of each sample along its arc, in [0, 1]
        starts = np.concatenate([[0], np.cumsum(n_samples)[:-1]])
        k = np.arange(len(owner)) - np.repeat(starts, n_samples)
        frac = k / np.repeat(n_samples - 1, n_samples)
        pts = a[owner] + frac[:, None] * (b - a)[owner]
        self._owner = owner
        self._tree = cKDTree(pts)
        self._a = a
        self._b = b

    @property
    def n_samples(self) -> int:
        return len(self._owner)

    def arcs_within(self, xy: np.ndarray, radius_m: float) -> list[np.ndarray]:
        """For each query point, the arc IDs whose segment comes within ``radius_m``.

        Returns unsorted unique arc IDs per point (exact distances are not
        returned here; use :meth:`distances` on the result).
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        hits = self._tree.query_ball_point(xy, r=radius_m + self.step_m / 2)
        out = []
        for i, h in enumerate(hits):
            if not h:
                out.append(np.empty(0, dtype=np.int64))
                continue
            arcs = np.unique(self._owner[h])
            d, _, _ = self.distances(np.broadcast_to(xy[i], (len(arcs), 2)), arcs)
            out.append(arcs[d <= radius_m])
        return out

    def distances(self, xy: np.ndarray, arc_ids: np.ndarray):
        """Exact point-to-segment (distance, t, closest_xy) for paired points and arcs."""
        arc_ids = np.asarray(arc_ids)
        return point_segment_distance(xy, self._a[arc_ids], self._b[arc_ids])
