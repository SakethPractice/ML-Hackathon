"""Expand the one-arc-per-observation Viterbi output into a connected directed route.

Between consecutive decoded states s_i = (arc a, f_i) and s_j = (arc b, f_j) of the *same*
Viterbi segment:

* same arc, forward movement      -> nothing inserted, arc not repeated;
* different arcs                  -> insert the arcs of the directed shortest node path
                                      to(a) -> from(b) (none if to(a) == from(b));
* same arc, "backward" movement   -> only via a genuine directed cycle (arcs inserted,
                                      then the arc is repeated). No path -> route is split.

No connecting road is ever invented: if no directed path exists within the search
radius, the route is split and a gap boundary is recorded. Distinct Viterbi segments are
never joined; each boundary keeps the reason that caused it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .config import DecoderConfig
from .network import DirectedRoadNetwork
from .transitions import is_feasible, search_bound_m, state_pair_distance
from .types import DecodedState


@dataclass
class RouteReconstruction:
    segments: list[list[int]]                 # arc ids per continuous route segment
    arc_ids: list[int]                        # flat concatenation of ``segments``
    geometry: list[list[list[float]]]         # one [lat, lon] polyline per segment
    gap_boundaries: list[dict[str, Any]] = field(default_factory=list)
    n_inserted_arcs: int = 0                  # intermediate arcs (not matched states)


def _connect(prev: DecodedState, nxt: DecodedState, network: DirectedRoadNetwork,
             config: DecoderConfig) -> Optional[tuple[str, list[int]]]:
    """Reconstruct only links that satisfy the HMM's hard feasibility rules.

    Without timestamps, still require directed reachability (legacy direct caller).
    """
    tcfg = config.transition
    if prev.timestamp is not None and nxt.timestamp is not None:
        dt = float(nxt.timestamp - prev.timestamp)
        if dt <= 0:
            return None
        bound = search_bound_m(dt, tcfg)
        d = state_pair_distance(network, prev.arc_id, prev.fraction, nxt.arc_id,
                                nxt.fraction, bound, tcfg.backward_tolerance_m)
        if not is_feasible(d, dt, tcfg):
            return None
    else:
        bound = tcfg.max_search_distance_m
    if prev.arc_id == nxt.arc_id:
        length = float(network.arc_length[prev.arc_id])
        if nxt.fraction * length >= prev.fraction * length - tcfg.backward_tolerance_m:
            return "stay", []
    u, v = network.to_node(prev.arc_id), network.from_node(nxt.arc_id)
    if u == v:
        return "move", []
    nodes = network.shortest_node_path(u, v, bound)
    if nodes is None:
        return None
    arcs = network.path_arc_ids(nodes)
    if len(arcs) > config.reconstruction.max_intermediate_arcs:
        return None
    return "move", arcs


def reconstruct_route(decoded_segments: Sequence[Sequence[DecodedState]],
                      network: DirectedRoadNetwork, config: DecoderConfig | None = None,
                      break_reasons: Sequence[str] = ()) -> RouteReconstruction:
    """Build the expanded route. ``break_reasons[k]`` explains the boundary between
    decoded segment k and k+1 (as produced by Viterbi)."""
    config = config or DecoderConfig()
    chunks: list[dict[str, Any]] = []
    inserted = 0

    for k, seg in enumerate(decoded_segments):
        if not seg:
            continue
        reason_before = (break_reasons[k - 1] if 0 < k <= len(break_reasons) else "segment_break") if k > 0 else None
        arcs = [seg[0].arc_id]
        first_obs = seg[0].observation_idx
        for prev, nxt in zip(seg[:-1], seg[1:]):
            link = _connect(prev, nxt, network, config)
            if link is None:
                chunks.append({"arcs": arcs, "first": first_obs, "last": prev.observation_idx,
                               "reason": reason_before})
                reason_before = "no_path_in_reconstruction"
                arcs, first_obs = [nxt.arc_id], nxt.observation_idx
                continue
            action, path = link
            if action == "stay":
                continue
            if config.reconstruction.insert_intermediate:
                arcs.extend(path)
                inserted += len(path)
            arcs.append(nxt.arc_id)
        chunks.append({"arcs": arcs, "first": first_obs, "last": seg[-1].observation_idx,
                       "reason": reason_before})

    segments = [c["arcs"] for c in chunks]
    flat: list[int] = []
    boundaries: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        if i > 0:
            boundaries.append({"after_observation_idx": chunks[i - 1]["last"],
                               "before_observation_idx": c["first"],
                               "reason": c["reason"] or "segment_break",
                               "expanded_route_index": len(flat)})
        flat.extend(c["arcs"])
    geometry = [network.route_polyline(s) for s in segments]
    return RouteReconstruction(segments, flat, geometry, boundaries, inserted)
