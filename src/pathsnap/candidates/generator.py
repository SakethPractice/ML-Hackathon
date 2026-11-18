"""Candidate road arcs for each GPS observation.

For observation ``i`` the candidates are the arcs whose point-to-segment
distance to the GPS fix is at most ``radius_m``, sorted by distance and cut
to the ``top_k`` closest. Ties are broken by arc ID so the output is fully
deterministic. Both directions of a two-way street have the same distance,
so they are always kept or dropped together unless the cut falls between
them.

Fallback when nothing is inside ``radius_m``: the radius is multiplied by
``fallback_factor`` repeatedly, up to ``max_radius_m``. The radius actually
used is stored in ``radius_used_m`` so later stages can see that the fix was
far from every road. If even ``max_radius_m`` finds nothing the observation
is returned with no candidates and is listed in ``CandidateSet.empty_obs``;
downstream code must handle that case instead of inventing an arc.

The generator never looks at the reference route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.graph.road_graph import RoadGraph
from pathsnap.graph.spatial_index import ArcIndex

CANDIDATE_COLUMNS = [
    "obs_id", "arc_id", "rank", "distance_m", "t_along", "proj_x", "proj_y", "radius_used_m",
]


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.0
    top_k: int = 8
    fallback_factor: float = 2.0
    max_radius_m: float = 400.0
    index_step_m: float = 25.0

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class CandidateSet:
    table: pd.DataFrame  # long format, one row per (obs_id, arc_id); see CANDIDATE_COLUMNS
    n_obs: int
    config: CandidateConfig

    @property
    def empty_obs(self) -> np.ndarray:
        has = np.zeros(self.n_obs, dtype=bool)
        has[self.table["obs_id"].to_numpy()] = True
        return np.flatnonzero(~has)

    def for_obs(self, obs_id: int) -> pd.DataFrame:
        t = self.table
        return t[t["obs_id"] == obs_id]

    def contains(self, obs_ids, arc_ids) -> np.ndarray:
        """Vectorised membership test: is ``arc_ids[j]`` a candidate of ``obs_ids[j]``?"""
        t = self.table
        key = set(zip(t["obs_id"].to_numpy().tolist(), t["arc_id"].to_numpy().tolist()))
        return np.array([(int(o), int(a)) in key for o, a in zip(obs_ids, arc_ids)], dtype=bool)


class CandidateGenerator:
    def __init__(self, graph: RoadGraph, config: CandidateConfig | None = None, index: ArcIndex | None = None):
        self.graph = graph
        self.config = config or CandidateConfig()
        self.index = index or ArcIndex(graph, step_m=self.config.index_step_m)

    def generate(self, track_xy: np.ndarray) -> CandidateSet:
        cfg = self.config
        track_xy = np.atleast_2d(np.asarray(track_xy, dtype=float))
        n = len(track_xy)

        obs_parts, arc_parts, rad_parts = [], [], []
        todo = np.arange(n)
        radius = cfg.radius_m
        while len(todo):
            found = self.index.arcs_within(track_xy[todo], radius)
            sizes = np.fromiter((len(f) for f in found), dtype=np.int64, count=len(found))
            ok = sizes > 0
            if ok.any():
                obs_parts.append(np.repeat(todo[ok], sizes[ok]))
                arc_parts.append(np.concatenate([f for f in found if len(f)]))
                rad_parts.append(np.full(int(sizes[ok].sum()), radius))
            todo = todo[~ok]
            if radius >= cfg.max_radius_m:
                break
            radius = min(radius * cfg.fallback_factor, cfg.max_radius_m)

        if not obs_parts:
            return CandidateSet(pd.DataFrame({c: [] for c in CANDIDATE_COLUMNS}), n, cfg)

        obs = np.concatenate(obs_parts)
        arcs = np.concatenate(arc_parts)
        rad = np.concatenate(rad_parts)
        dist, t, closest = self.index.distances(track_xy[obs], arcs)

        order = np.lexsort((arcs, dist, obs))
        obs, arcs, rad, dist, t, closest = obs[order], arcs[order], rad[order], dist[order], t[order], closest[order]
        # rank within each observation
        first = np.r_[True, obs[1:] != obs[:-1]]
        group_start = np.maximum.accumulate(np.where(first, np.arange(len(obs)), 0))
        rank = np.arange(len(obs)) - group_start
        keep = rank < cfg.top_k

        table = pd.DataFrame({
            "obs_id": obs[keep],
            "arc_id": arcs[keep],
            "rank": rank[keep],
            "distance_m": dist[keep],
            "t_along": t[keep],
            "proj_x": closest[keep, 0],
            "proj_y": closest[keep, 1],
            "radius_used_m": rad[keep],
        })
        return CandidateSet(table.reset_index(drop=True), n, cfg)
