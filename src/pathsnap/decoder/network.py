"""Directed road network with metre-based geometry and cached bounded shortest paths.

Arc IDs and node IDs are *trip-local* row indices, exactly as in the ``.arcs`` /
``.nodes`` files. Parallel arcs are allowed as *candidates*; for node-to-node
path search only the shortest parallel arc (lowest arc id on ties) is used.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Optional, Sequence

import networkx as nx
import numpy as np

EARTH_RADIUS_M = 6371008.8
_M_PER_DEG = EARTH_RADIUS_M * math.pi / 180.0


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class DirectedRoadNetwork:
    def __init__(self, node_lonlat, arcs, arc_length_m=None, name: str = "",
                 cache_size: int = 4096, bucket_m: float = 250.0) -> None:
        nodes = np.asarray(node_lonlat, dtype=float)
        arcs_arr = np.asarray(arcs)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("node_lonlat must have shape (n_nodes, 2) = (lon, lat)")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("node coordinates must be finite")
        if arcs_arr.ndim != 2 or arcs_arr.shape[1] != 2:
            raise ValueError("arcs must have shape (n_arcs, 2) = (from_node, to_node)")
        if not np.issubdtype(arcs_arr.dtype, np.integer):
            if arcs_arr.size and not np.all(arcs_arr == np.floor(arcs_arr)):
                raise ValueError("arcs must contain integer node ids")
        arcs_arr = arcs_arr.astype(np.int64)
        if arcs_arr.size and (arcs_arr.min() < 0 or arcs_arr.max() >= len(nodes)):
            raise ValueError("arcs reference node ids outside the node table")
        self.name = name
        self.node_lonlat = nodes
        self.arcs = arcs_arr
        if arc_length_m is None:
            arc_length_m = haversine_m(nodes[arcs_arr[:, 0], 0], nodes[arcs_arr[:, 0], 1],
                                       nodes[arcs_arr[:, 1], 0], nodes[arcs_arr[:, 1], 1])
        lengths = np.asarray(arc_length_m, dtype=float)
        if lengths.shape != (len(arcs_arr),) or np.any(~np.isfinite(lengths)) or np.any(lengths < 0):
            raise ValueError("arc_length_m must be a finite non-negative vector of length n_arcs")
        self.arc_length = lengths

        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(nodes)))
        for aid, (u, v) in enumerate(arcs_arr.tolist()):
            if u == v:
                continue  # self-loops never help a shortest path
            length = float(lengths[aid])
            if graph.has_edge(u, v):
                if length < graph[u][v]["length"]:
                    graph[u][v].update(length=length, arc_id=aid)
            else:
                graph.add_edge(u, v, length=length, arc_id=aid)
        self.graph = graph

        self._cache: "OrderedDict[int, tuple]" = OrderedDict()
        self._cache_size = int(cache_size)
        self._bucket = float(bucket_m)
        self._lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0

    # ---- basic accessors ------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.node_lonlat)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    def has_arc(self, arc_id: int) -> bool:
        return 0 <= int(arc_id) < self.n_arcs

    def from_node(self, arc_id: int) -> int:
        return int(self.arcs[arc_id, 0])

    def to_node(self, arc_id: int) -> int:
        return int(self.arcs[arc_id, 1])

    # ---- geometry (metres) ---------------------------------------------
    def project_points(self, lon: float, lat: float, arc_ids: np.ndarray):
        """Project one GPS point onto each arc (straight segment between its nodes).

        Returns (fraction in [0,1] along arc, perpendicular distance in metres),
        computed in a local equirectangular frame centred on the GPS point.
        """
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        a = self.arcs[arc_ids]
        cos_lat = math.cos(math.radians(lat))
        x1 = (self.node_lonlat[a[:, 0], 0] - lon) * cos_lat * _M_PER_DEG
        y1 = (self.node_lonlat[a[:, 0], 1] - lat) * _M_PER_DEG
        x2 = (self.node_lonlat[a[:, 1], 0] - lon) * cos_lat * _M_PER_DEG
        y2 = (self.node_lonlat[a[:, 1], 1] - lat) * _M_PER_DEG
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = np.divide(-(x1 * dx + y1 * dy), seg2, out=np.zeros_like(seg2), where=seg2 > 0)
        t = np.clip(t, 0.0, 1.0)
        dist = np.hypot(x1 + t * dx, y1 + t * dy)
        return t, dist

    def route_polyline(self, arc_ids: Sequence[int]) -> list[list[float]]:
        """``[[lat, lon], ...]`` through consecutive arcs (Folium order)."""
        nodes: list[int] = []
        last = None
        for a in arc_ids:
            u, v = int(self.arcs[a, 0]), int(self.arcs[a, 1])
            if last != u:
                nodes.append(u)
            nodes.append(v)
            last = v
        return [[float(self.node_lonlat[n, 1]), float(self.node_lonlat[n, 0])] for n in nodes]

    # ---- bounded shortest paths (cached) --------------------------------
    def reachable(self, source: int, need_m: float) -> dict:
        """Node -> shortest directed distance (m) for nodes within >= ``need_m`` of source.

        One Dijkstra per source is cached and reused across all observation pairs;
        it is recomputed only when a larger radius is required.
        """
        with self._lock:
            entry = self._cache.get(source)
            if entry is not None and entry[0] >= need_m:
                self._cache.move_to_end(source)
                self.cache_hits += 1
                return entry[1]
            self.cache_misses += 1
            cutoff = max(self._bucket, math.ceil(need_m / self._bucket) * self._bucket)
            dist, paths = nx.single_source_dijkstra(self.graph, source, cutoff=cutoff, weight="length")
            self._cache[source] = (cutoff, dist, paths)
            self._cache.move_to_end(source)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return dist

    def shortest_node_path(self, u: int, v: int, cutoff_m: float) -> Optional[list[int]]:
        """Directed shortest node path u -> v within ``cutoff_m`` or None if unreachable."""
        if u == v:
            return [u]
        self.reachable(u, cutoff_m)
        with self._lock:
            entry = self._cache.get(u)
            path = entry[2].get(v) if entry is not None else None
        if not path:
            return None
        with self._lock:
            d = self._cache.get(u, (None, {}, {}))[1].get(v, float("inf"))
        return list(path) if d <= cutoff_m + 1e-9 else None

    def path_arc_ids(self, node_path: Sequence[int]) -> list[int]:
        return [int(self.graph[a][b]["arc_id"]) for a, b in zip(node_path[:-1], node_path[1:])]
