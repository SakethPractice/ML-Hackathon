"""Directed-network transition scores between candidate states of consecutive observations.

Network distance between state i = (arc a, fraction f_i) and state j = (arc b, fraction f_j):

  same arc, forward   :  d = max(0, (f_j - f_i) * L_a)          if f_j*L_a >= f_i*L_a - tol
  otherwise           :  d = (1 - f_i) * L_a  +  SP(to(a) -> from(b))  +  f_j * L_b

where SP is the directed shortest-path length in metres (inf if unreachable within the
search radius). A same-arc "backward" move therefore needs a genuine directed cycle.

Log transition score (-inf = impossible):

  log T = - |d - d_gps| / beta  -  w * max(0, v - v_soft) / v_soft ,   v = d / dt
  T = -inf  if d is inf  or  v > v_max
"""
from __future__ import annotations

import numpy as np

from .config import TransitionConfig
from .network import DirectedRoadNetwork, haversine_m
from .types import ObservationStates


def network_distance_matrix(network: DirectedRoadNetwork, prev: ObservationStates,
                            cur: ObservationStates, need_m: float,
                            backward_tolerance_m: float) -> np.ndarray:
    """(n_prev, n_cur) directed network distances in metres (inf = unreachable)."""
    len_prev = network.arc_length[prev.arc_ids]
    len_cur = network.arc_length[cur.arc_ids]
    off_prev = prev.fractions * len_prev
    off_cur = cur.fractions * len_cur
    rem = len_prev - off_prev

    end_nodes = network.arcs[prev.arc_ids, 1]
    start_nodes = network.arcs[cur.arc_ids, 0]
    uniq_u, inv_u = np.unique(end_nodes, return_inverse=True)
    uniq_v, inv_v = np.unique(start_nodes, return_inverse=True)
    table = np.full((len(uniq_u), len(uniq_v)), np.inf)
    for a, u in enumerate(uniq_u.tolist()):
        reach = network.reachable(u, need_m)       # one cached Dijkstra per unique source
        for b, v in enumerate(uniq_v.tolist()):
            if u == v:
                table[a, b] = 0.0
            else:
                d = reach.get(v)
                if d is not None and d <= need_m + 1e-9:
                    table[a, b] = d
    general = rem[:, None] + table[inv_u.ravel()][:, inv_v.ravel()] + off_cur[None, :]

    same = prev.arc_ids[:, None] == cur.arc_ids[None, :]
    forward = off_cur[None, :] >= off_prev[:, None] - backward_tolerance_m
    same_forward = same & forward
    return np.where(same_forward, np.maximum(off_cur[None, :] - off_prev[:, None], 0.0), general)


def calculate_transition_scores(prev: ObservationStates, cur: ObservationStates,
                                network: DirectedRoadNetwork, config: TransitionConfig):
    """Return ``(log_transition (n_prev, n_cur), network_distance_m)``."""
    dt = float(cur.timestamp - prev.timestamp)
    if dt <= 0:
        raise ValueError("calculate_transition_scores requires strictly increasing timestamps")
    d_gps = float(haversine_m(prev.lon, prev.lat, cur.lon, cur.lat))
    need = search_bound_m(dt, config)
    d_net = network_distance_matrix(network, prev, cur, need, config.backward_tolerance_m)

    with np.errstate(invalid="ignore", over="ignore"):
        speed = d_net / dt
        finite = np.isfinite(d_net)
        safe = np.where(finite, d_net, 0.0)
        mismatch = np.abs(safe - d_gps)
        over_soft = np.maximum(safe / dt - config.soft_speed_mps, 0.0)
        log_t = -mismatch / config.beta_m - config.speed_penalty_weight * over_soft / config.soft_speed_mps
    infeasible = ~finite | (speed > config.max_speed_mps)
    return np.where(infeasible, -np.inf, log_t), d_net


def search_bound_m(dt: float, config: TransitionConfig) -> float:
    """Dijkstra search distance cap for this time gap (metres)."""
    return min(config.max_search_distance_m, config.max_speed_mps * dt + config.speed_slack_m)


def is_feasible(distance_m: float, dt: float, config: TransitionConfig) -> bool:
    """Shared hard transition constraint (directed distance and implied speed)."""
    return bool(np.isfinite(distance_m) and dt > 0 and distance_m / dt <= config.max_speed_mps)


def state_pair_distance(network: DirectedRoadNetwork, prev_arc: int, prev_fraction: float,
                        cur_arc: int, cur_fraction: float, need_m: float,
                        backward_tolerance_m: float) -> float:
    """Single state-pair distance using exactly the decoder's distance calculation."""
    def single(arc: int, fraction: float) -> ObservationStates:
        zero = np.zeros(1)
        return ObservationStates(0, 0.0, 0.0, 0.0, np.array([arc], dtype=np.int64),
                                 np.array([fraction], dtype=float), zero, zero, zero)
    return float(network_distance_matrix(network, single(prev_arc, prev_fraction),
                                         single(cur_arc, cur_fraction), need_m,
                                         backward_tolerance_m)[0, 0])
