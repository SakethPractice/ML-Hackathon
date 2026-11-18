"""HMM formulation for map matching + public ``decode_trip`` entry point.

Hidden states     : candidate directed arcs of each GPS observation.
Observations      : GPS (lon, lat, timestamp).
Emission scores   : discriminative pseudo-emissions from the binary match classifier,
                    log q_t(a) with q_t(a) = clip(p_a) / sum_b clip(p_b). These are
                    compatibility scores, NOT a generative P(GPS | road).
Transition scores : directed network distance vs. GPS displacement and speed
                    (see :mod:`pathsnap.decoder.transitions`).
No extra distance-likelihood is added to the emission: distance is already a
classifier feature (avoids double counting).

Reference routes / labels are never read here; inference uses an explicit column subset.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import DecoderConfig, EmissionConfig
from .network import DirectedRoadNetwork
from .route_reconstruction import reconstruct_route
from .transitions import calculate_transition_scores
from .types import DecodedState, ObservationStates, TripDecodeResult, normalise_trip_id
from .viterbi import viterbi_decode

logger = logging.getLogger(__name__)

_OBS_COLS = ("observation_idx", "lon", "lat", "timestamp")
_CAND_COLS = ("observation_idx", "arc_id", "match_probability")
_LEAKY_COLS = ("label", "y_true", "reference_arc_id", "ref_arc_id", "is_reference", "reference_route")


# ---------------------------------------------------------------- emissions
def clean_match_scores(scores, epsilon: float) -> tuple[np.ndarray, int]:
    """Clip scores to [eps, 1-eps]; NaN/inf/out-of-[0,1] are treated as zero evidence (eps).

    Returns (cleaned scores, number of invalid inputs)."""
    p = np.asarray(scores, dtype=float).ravel()
    invalid = ~np.isfinite(p) | (p < 0.0) | (p > 1.0)
    p = np.where(invalid, 0.0, p)
    return np.clip(p, epsilon, 1.0 - epsilon), int(invalid.sum())


def calculate_emission_scores(match_scores, config: EmissionConfig | None = None) -> np.ndarray:
    """Log pseudo-emission scores for the candidates of ONE observation."""
    cfg = config or EmissionConfig()
    p, _ = clean_match_scores(match_scores, cfg.epsilon)
    if p.size == 0:
        return p
    if cfg.mode == "normalised":
        return np.log(p / p.sum())
    if cfg.mode == "logit_normalised":
        z = (np.log(p) - np.log1p(-p)) / cfg.temperature
        z = z - z.max()
        return z - np.log(np.exp(z).sum())
    return np.log(p)  # raw_log


# ------------------------------------------------------------ preparation
@dataclass
class PreparedTrip:
    trip_id: str
    observation_indices: list[int]
    observation_status: dict[int, str]       # idx -> candidate_ready | no_candidates | invalid_*
    states: list[ObservationStates]          # only observations that are decodable
    skipped_before: list[list[tuple[int, str]]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _filter_trip(df: pd.DataFrame, tid: str) -> pd.DataFrame:
    if "trip_id" in df.columns:
        return df.loc[df["trip_id"].map(normalise_trip_id) == tid]
    return df


def prepare_trip(trip_id: Any, observations: pd.DataFrame, candidate_predictions: pd.DataFrame,
                 road_graph: DirectedRoadNetwork, config: DecoderConfig | None = None) -> PreparedTrip:
    """Validate inputs and build per-observation candidate state sets."""
    config = config or DecoderConfig()
    tid = normalise_trip_id(trip_id)
    missing = [c for c in _OBS_COLS if c not in observations.columns]
    if missing:
        raise ValueError(f"observations missing columns: {missing}")
    missing = [c for c in _CAND_COLS if c not in candidate_predictions.columns]
    if missing:
        raise ValueError(f"candidate_predictions missing columns: {missing}")
    leaky = [c for c in _LEAKY_COLS if c in candidate_predictions.columns]
    if leaky:
        logger.warning("Ignoring label-like columns in candidate_predictions (never used for inference): %s", leaky)

    obs = _filter_trip(observations, tid)[list(_OBS_COLS)].copy()
    if obs.empty:
        raise ValueError(f"no observations supplied for trip {tid}")
    for c in _OBS_COLS:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")
    if obs["observation_idx"].isna().any():
        raise ValueError("observation_idx must be present for every observation")
    if obs["observation_idx"].duplicated().any():
        raise ValueError("duplicate observation_idx values")
    obs["observation_idx"] = obs["observation_idx"].astype(np.int64)
    obs = obs.sort_values("observation_idx", kind="mergesort")

    keep = list(_CAND_COLS) + (["proj_fraction"] if "proj_fraction" in candidate_predictions.columns else [])
    cand = _filter_trip(candidate_predictions, tid)[keep].copy()
    for c in keep:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    n_rows = len(cand)
    arc_ok = (cand["observation_idx"].notna() & cand["arc_id"].notna()
              & (cand["arc_id"] == np.floor(cand["arc_id"]))
              & (cand["arc_id"] >= 0) & (cand["arc_id"] < road_graph.n_arcs))
    n_bad_arc = int((~arc_ok).sum())
    cand = cand.loc[arc_ok]
    orphan = ~cand["observation_idx"].isin(obs["observation_idx"])
    n_orphan = int(orphan.sum())
    cand = cand.loc[~orphan].copy()
    cand["observation_idx"] = cand["observation_idx"].astype(np.int64)
    cand["arc_id"] = cand["arc_id"].astype(np.int64)
    before = len(cand)
    cand = (cand.sort_values("match_probability", ascending=False, na_position="last", kind="mergesort")
                .drop_duplicates(subset=["observation_idx", "arc_id"], keep="first"))
    n_dup = before - len(cand)
    groups = {int(k): g for k, g in cand.groupby("observation_idx", sort=False)}

    status: dict[int, str] = {}
    states: list[ObservationStates] = []
    skipped_before: list[list[tuple[int, str]]] = []
    pending: list[tuple[int, str]] = []
    last_ts = -np.inf
    n_invalid_scores = 0
    for row in obs.itertuples(index=False):
        idx = int(row.observation_idx)
        lon, lat, ts = float(row.lon), float(row.lat), float(row.timestamp)
        if not (np.isfinite(lon) and np.isfinite(lat) and np.isfinite(ts)) or abs(lat) > 90 or abs(lon) > 180:
            status[idx] = "invalid_observation"
            pending.append((idx, "invalid_observation"))
            continue
        if ts <= last_ts:
            status[idx] = "invalid_timestamp"     # duplicate / non-increasing: excluded, not decoded
            pending.append((idx, "invalid_timestamp"))
            continue
        last_ts = ts
        g = groups.get(idx)
        if g is None or g.empty:
            status[idx] = "no_candidates"
            pending.append((idx, "no_candidates"))
            continue
        g = g.sort_values("arc_id", kind="mergesort")
        arc_ids = g["arc_id"].to_numpy(dtype=np.int64)
        raw = g["match_probability"].to_numpy(dtype=float)
        cleaned, n_inv = clean_match_scores(raw, config.emission.epsilon)
        n_invalid_scores += n_inv
        frac, dist = road_graph.project_points(lon, lat, arc_ids)
        if "proj_fraction" in g.columns:
            given = g["proj_fraction"].to_numpy(dtype=float)
            ok = np.isfinite(given) & (given >= 0.0) & (given <= 1.0)
            frac = np.where(ok, given, frac)
        states.append(ObservationStates(idx, lon, lat, ts, arc_ids, frac, dist, cleaned,
                                        calculate_emission_scores(raw, config.emission)))
        status[idx] = "candidate_ready"
        skipped_before.append(pending)
        pending = []

    counts = pd.Series(list(status.values())).value_counts().to_dict()
    diag = {
        "n_observations": int(len(obs)),
        "n_candidate_rows_input": int(n_rows),
        "n_dropped_invalid_arc": n_bad_arc,
        "n_dropped_orphan_candidates": n_orphan,
        "n_dropped_duplicate_candidates": int(n_dup),
        "n_invalid_scores_replaced": int(n_invalid_scores),
        "observation_status_counts": {str(k): int(v) for k, v in counts.items()},
        "mean_candidates_per_decodable_observation":
            float(np.mean([len(s) for s in states])) if states else 0.0,
    }
    return PreparedTrip(tid, [int(i) for i in obs["observation_idx"]], status, states, skipped_before, diag)


# ---------------------------------------------------------------- decoding
def decode_prepared(prepared: PreparedTrip, road_graph: DirectedRoadNetwork,
                    config: DecoderConfig | None = None) -> TripDecodeResult:
    """Viterbi decode + route reconstruction for an already prepared trip."""
    config = config or DecoderConfig()
    t0 = time.perf_counter()
    hits0, miss0 = road_graph.cache_hits, road_graph.cache_misses
    states = prepared.states

    def transition_fn(i: int, j: int) -> np.ndarray:
        return calculate_transition_scores(states[i], states[j], road_graph, config.transition)[0]

    def break_fn(i: int, j: int) -> Optional[str]:
        if states[j].timestamp - states[i].timestamp > config.gaps.max_gap_seconds:
            return "large_time_gap"
        skipped = prepared.skipped_before[j] if j < len(prepared.skipped_before) else []
        if config.gaps.missing_candidate_policy == "split" and any(reason == "no_candidates" for _, reason in skipped):
            return "missing_candidates"
        if len(skipped) > config.gaps.max_bridged_observations:
            return "too_many_skipped_observations"
        return None

    vit = viterbi_decode([s.log_emissions for s in states], transition_fn, break_fn)
    t_viterbi = time.perf_counter()

    decoded_segments: list[list[DecodedState]] = []
    matched: dict[int, int] = {}
    for seg in vit.segments:
        cur: list[DecodedState] = []
        for pos, k in zip(seg.positions, seg.states):
            s = states[pos]
            ds = DecodedState(s.observation_idx, int(s.arc_ids[k]), float(s.fractions[k]), s.timestamp)
            cur.append(ds)
            matched[ds.observation_idx] = ds.arc_id
        decoded_segments.append(cur)
    route = reconstruct_route(decoded_segments, road_graph, config, [r for _, _, r in vit.breaks])

    skipped_all = [{"observation_idx": idx, "reason": prepared.observation_status[idx]}
                   for idx in prepared.observation_indices if idx not in matched]
    bridged = []
    for seg in vit.segments:
        for before, after in zip(seg.positions[:-1], seg.positions[1:]):
            skips = prepared.skipped_before[after] if after < len(prepared.skipped_before) else []
            if skips:
                bridged.append({"after_observation_idx": states[before].observation_idx,
                                "before_observation_idx": states[after].observation_idx,
                                "skipped_observation_indices": [idx for idx, _ in skips],
                                "reasons": [reason for _, reason in skips]})
    for boundary in route.gap_boundaries:
        lo, hi = boundary["after_observation_idx"], boundary["before_observation_idx"]
        boundary["skipped_observation_indices"] = [idx for idx in prepared.observation_indices
                                                    if lo < idx < hi and idx not in matched]
    obs_status = [("decoded" if i in matched else prepared.observation_status[i])
                  for i in prepared.observation_indices]
    n_unresolved = sum(1 for s in obs_status if s != "decoded")
    if not matched:
        status = "failed"
    elif len(vit.segments) > 1:
        status = "split"
    elif n_unresolved:
        status = "success_with_unresolved"
    else:
        status = "success"

    runtime = time.perf_counter() - t0
    diagnostics = {
        **prepared.diagnostics,
        "viterbi": vit.diagnostics,
        "skipped_observations": skipped_all,
        "bridged_skips": bridged,
        "missing_candidate_policy": config.gaps.missing_candidate_policy,
        "n_decoded_observations": len(matched),
        "n_unresolved_observations": n_unresolved,
        "n_route_segments": len(route.segments),
        "n_inserted_intermediate_arcs": route.n_inserted_arcs,
        "viterbi_seconds": t_viterbi - t0,
        "reconstruction_seconds": time.perf_counter() - t_viterbi,
        "shortest_path_cache_hits": road_graph.cache_hits - hits0,
        "shortest_path_cache_misses": road_graph.cache_misses - miss0,
        "emission_mode": config.emission.mode,
    }
    return TripDecodeResult(
        trip_id=prepared.trip_id, status=status,
        observation_indices=list(prepared.observation_indices),
        matched_arc_ids=[matched.get(i) for i in prepared.observation_indices],
        observation_status=obs_status,
        expanded_route_arc_ids=route.arc_ids, route_segments=route.segments,
        route_geometry=route.geometry, gap_boundaries=route.gap_boundaries,
        diagnostics=diagnostics, runtime_seconds=runtime)


def decode_trip(trip_id: Any, observations: pd.DataFrame, candidate_predictions: pd.DataFrame,
                road_graph: DirectedRoadNetwork, config: DecoderConfig | None = None) -> TripDecodeResult:
    """Main entry point for Member 4 / FastAPI.

    observations          : columns observation_idx, lon, lat, timestamp (+ optional trip_id)
    candidate_predictions : columns observation_idx, arc_id, match_probability
                            (+ optional trip_id, proj_fraction); other columns are ignored
    road_graph            : :class:`DirectedRoadNetwork` of THIS trip
    """
    t0 = time.perf_counter()
    config = config or DecoderConfig()
    prepared = prepare_trip(trip_id, observations, candidate_predictions, road_graph, config)
    result = decode_prepared(prepared, road_graph, config)
    result.runtime_seconds = time.perf_counter() - t0   # include preparation
    return result
