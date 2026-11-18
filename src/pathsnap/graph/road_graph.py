"""Directed road graph for one trip.

Arcs keep the IDs from the ``.arcs`` file (row index) and nodes keep the IDs
from the ``.nodes`` file. Nothing is renumbered, so an arc ID produced here
can be compared directly with an arc ID in a ``.route`` file.

The graph is directed: an arc ``u -> v`` allows travel from ``u`` to ``v``
only. A two-way street appears in the data as two arcs, one per direction.
Arc ``a`` can be followed by arc ``b`` when ``to_node(a) == from_node(b)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.data.loader import Trip
from pathsnap.graph.geo import LocalProjection, bearing_deg, haversine_m


@dataclass
class RoadGraph:
    node_lonlat: np.ndarray  # (n_nodes, 2)
    node_xy: np.ndarray  # (n_nodes, 2) metres in ``projection``
    arc_from: np.ndarray  # (n_arcs,)
    arc_to: np.ndarray  # (n_arcs,)
    arc_length_m: np.ndarray  # haversine length
    arc_bearing_deg: np.ndarray  # 0 = north, clockwise
    projection: LocalProjection
    # CSR style lists: arcs leaving / entering each node
    _out_ptr: np.ndarray
    _out_arcs: np.ndarray
    _in_ptr: np.ndarray
    _in_arcs: np.ndarray

    @classmethod
    def from_frames(cls, nodes: pd.DataFrame, arcs: pd.DataFrame, projection: LocalProjection | None = None):
        node_lonlat = nodes[["lon", "lat"]].to_numpy(dtype=float)
        arc_from = arcs["from_node"].to_numpy(dtype=np.int64)
        arc_to = arcs["to_node"].to_numpy(dtype=np.int64)
        n = len(node_lonlat)
        if len(arc_from) and (min(arc_from.min(), arc_to.min()) < 0 or max(arc_from.max(), arc_to.max()) >= n):
            raise ValueError("arcs reference node IDs that are not in the nodes file")

        if projection is None:
            projection = LocalProjection.centred_on(node_lonlat[:, 0], node_lonlat[:, 1])
        node_xy = projection.to_xy(node_lonlat[:, 0], node_lonlat[:, 1])

        a = node_lonlat[arc_from]
        b = node_lonlat[arc_to]
        length = haversine_m(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        d = node_xy[arc_to] - node_xy[arc_from]
        bearing = bearing_deg(d[:, 0], d[:, 1])

        out_ptr, out_arcs = _group_by(arc_from, n)
        in_ptr, in_arcs = _group_by(arc_to, n)
        return cls(node_lonlat, node_xy, arc_from, arc_to, length, bearing, projection,
                   out_ptr, out_arcs, in_ptr, in_arcs)

    @classmethod
    def from_trip(cls, trip: Trip, projection: LocalProjection | None = None) -> "RoadGraph":
        return cls.from_frames(trip.nodes, trip.arcs, projection)

    # ------------------------------------------------------------------ basics
    @property
    def n_nodes(self) -> int:
        return len(self.node_xy)

    @property
    def n_arcs(self) -> int:
        return len(self.arc_from)

    def arc_xy(self, arc_ids) -> tuple[np.ndarray, np.ndarray]:
        arc_ids = np.asarray(arc_ids)
        return self.node_xy[self.arc_from[arc_ids]], self.node_xy[self.arc_to[arc_ids]]

    def arc_lonlat(self, arc_id: int) -> np.ndarray:
        """[[lon, lat] of from-node, [lon, lat] of to-node]."""
        return self.node_lonlat[[self.arc_from[arc_id], self.arc_to[arc_id]]]

    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    def in_arcs(self, node: int) -> np.ndarray:
        return self._in_arcs[self._in_ptr[node]:self._in_ptr[node + 1]]

    def successors(self, arc_id: int) -> np.ndarray:
        """Arcs that can directly follow ``arc_id``."""
        return self.out_arcs(self.arc_to[arc_id])

    def predecessors(self, arc_id: int) -> np.ndarray:
        return self.in_arcs(self.arc_from[arc_id])

    def follows(self, a, b) -> np.ndarray:
        """True where arc ``b`` can be driven directly after arc ``a``."""
        return self.arc_to[np.asarray(a)] == self.arc_from[np.asarray(b)]

    def node_degree(self) -> tuple[np.ndarray, np.ndarray]:
        """(out_degree, in_degree) for every node."""
        return np.diff(self._out_ptr), np.diff(self._in_ptr)

    def reverse_arc_exists(self) -> np.ndarray:
        """Per arc: is there also an arc going the opposite way between the same nodes?"""
        n = self.n_nodes
        key = self.arc_from * n + self.arc_to
        rev = self.arc_to * n + self.arc_from
        return np.isin(rev, key)

    # ------------------------------------------------------------ connectivity
    def _adjacency(self) -> csr_matrix:
        n = self.n_nodes
        data = np.ones(self.n_arcs, dtype=np.int8)
        return csr_matrix((data, (self.arc_from, self.arc_to)), shape=(n, n))

    def components(self, connection: str = "weak") -> np.ndarray:
        """Component label per node. ``connection`` is 'weak' or 'strong'."""
        _, labels = connected_components(self._adjacency(), directed=True, connection=connection)
        return labels

    def connectivity_summary(self) -> dict:
        out_deg, in_deg = self.node_degree()
        isolated = (out_deg + in_deg) == 0
        weak = self.components("weak")
        strong = self.components("strong")
        # only count components that contain at least one arc
        used = ~isolated
        weak_sizes = np.bincount(weak[used]) if used.any() else np.array([0])
        strong_sizes = np.bincount(strong[used]) if used.any() else np.array([0])
        weak_sizes = weak_sizes[weak_sizes > 0]
        strong_sizes = strong_sizes[strong_sizes > 0]
        n_used = int(used.sum())
        dead_end = used & (out_deg == 0)
        source_only = used & (in_deg == 0)
        return {
            "n_isolated_nodes": int(isolated.sum()),
            "n_weak_components": int(len(weak_sizes)),
            "largest_weak_component_share": float(weak_sizes.max() / n_used) if n_used else np.nan,
            "n_strong_components": int(len(strong_sizes)),
            "largest_strong_component_share": float(strong_sizes.max() / n_used) if n_used else np.nan,
            "n_dead_end_nodes": int(dead_end.sum()),
            "n_source_only_nodes": int(source_only.sum()),
            "share_arcs_two_way": float(self.reverse_arc_exists().mean()) if self.n_arcs else np.nan,
            "arc_length_median_m": float(np.median(self.arc_length_m)) if self.n_arcs else np.nan,
            "arc_length_max_m": float(self.arc_length_m.max()) if self.n_arcs else np.nan,
            "n_zero_length_arcs": int((self.arc_length_m == 0).sum()),
        }


def _group_by(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR pointer/array pair listing, for each key in [0, n), the row positions with that key."""
    order = np.argsort(keys, kind="stable")
    counts = np.bincount(keys, minlength=n)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    return ptr, order.astype(np.int64)
