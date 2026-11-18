"""Metric definitions (candidate-, observation- and route-level).

Trip-local arc IDs are NEVER used as global classes: Macro F1 and log loss are the
candidate-row *binary* versions; Top-1 compares trip-local arc ids within one trip.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from ..decoder.config import DecoderConfig
from ..decoder.network import DirectedRoadNetwork
from ..decoder.transitions import calculate_transition_scores
from ..decoder.types import ObservationStates


# ------------------------------------------------------------ candidate level
def candidate_level_metrics(y_true, y_pred, p=None, eps: float = 1e-15) -> dict[str, float]:
    """Binary accuracy / precision / recall / Macro F1 (mean of F1 for label 0 and 1) / log loss."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    nan = float("nan")
    if y_true.size == 0:
        return {"n_candidates": 0, "accuracy": nan, "precision": nan, "recall": nan,
                "macro_f1": nan, "log_loss": nan}
    out = {
        "n_candidates": int(y_true.size),
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "log_loss": nan,
    }
    if p is not None:
        out["log_loss"] = float(log_loss(y_true, np.clip(np.asarray(p, float), eps, 1 - eps), labels=[0, 1]))
    return out


def labelled_top1_accuracy(trip_ids, observation_ids, y_true, scores) -> tuple[float, int, int]:
    """CANDIDATE-TABLE Top-1 proxy used during tuning; NOT full observation-level accuracy.

    For every (trip, observation) that has at least one positive candidate, check that the
    highest-scoring candidate (ties: first row) is positive. Returns
    (accuracy, n_observations_with_positive, n_observations_without_positive).
    Observations without any positive candidate (possible candidate misses / unaligned) are
    excluded from the denominator here because the table alone cannot distinguish them.
    """
    df = pd.DataFrame({"t": np.asarray(trip_ids).astype(str), "o": np.asarray(observation_ids),
                       "y": np.asarray(y_true).astype(int), "s": np.asarray(scores, float)})
    has_pos = df.groupby(["t", "o"], sort=False)["y"].max()
    top = (df.sort_values(["t", "o", "s"], ascending=[True, True, False], kind="mergesort")
             .groupby(["t", "o"], sort=False).head(1).set_index(["t", "o"])["y"])
    mask = has_pos == 1
    n_pos, n_none = int(mask.sum()), int((~mask).sum())
    if n_pos == 0:
        return float("nan"), 0, n_none
    return float((top.loc[mask[mask].index] == 1).mean()), n_pos, n_none


# ----------------------------------------------------- reference alignment
def derive_alignment_from_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Alignment table from Member 2's candidate labels (evaluation only).

    exactly one positive -> aligned; several -> ambiguous; none -> unaligned
    (cannot be distinguished from a candidate miss from labels alone; prefer Member 2's
    explicit alignment table when available).
    """
    pos = df.loc[df["label"].astype(int) == 1].groupby(["trip_id", "observation_idx"])["arc_id"].agg(list)
    rows = []
    for (t, o), arcs in pos.items():
        uniq = sorted(set(int(a) for a in arcs))
        rows.append((t, int(o), uniq[0] if len(uniq) == 1 else np.nan,
                     "aligned" if len(uniq) == 1 else "ambiguous"))
    out = pd.DataFrame(rows, columns=["trip_id", "observation_idx", "ref_arc_id", "alignment_status"])
    allobs = df[["trip_id", "observation_idx"]].drop_duplicates()
    out = allobs.merge(out, on=["trip_id", "observation_idx"], how="left")
    out["alignment_status"] = out["alignment_status"].fillna("unaligned")
    return out


def observation_outcomes(observation_indices: Sequence[int], pred_by_obs: Mapping[int, Optional[int]],
                         candidate_arcs: Mapping[int, set], alignment: pd.DataFrame) -> pd.DataFrame:
    """Per-observation Top-1 outcome for ONE trip and ONE pipeline.

    Denominator = observations with alignment_status == 'aligned'. A reference arc that is
    absent from the candidate set counts as incorrect AND as a candidate miss; an
    observation the pipeline could not resolve also counts as incorrect.
    """
    amap = {int(r.observation_idx): (r.alignment_status, r.ref_arc_id) for r in alignment.itertuples(index=False)}
    rows = []
    for o in observation_indices:
        o = int(o)
        st, ref = amap.get(o, ("no_alignment", np.nan))
        pred = pred_by_obs.get(o)
        in_den = st == "aligned" and pd.notna(ref)
        ref_i = int(ref) if in_den else None
        miss = bool(in_den and ref_i not in candidate_arcs.get(o, set()))
        rows.append({"observation_idx": o, "alignment_status": st,
                     "ref_arc_id": ref_i, "pred_arc_id": pred,
                     "in_denominator": in_den, "candidate_miss": miss,
                     "unresolved": bool(in_den and pred is None),
                     "correct": bool(in_den and pred is not None and pred == ref_i)})
    return pd.DataFrame(rows)


def summarise_outcomes(outcomes: pd.DataFrame) -> dict[str, float]:
    if outcomes.empty:
        return {"n_observations": 0, "n_aligned": 0, "n_ambiguous_excluded": 0, "n_unaligned_excluded": 0,
                "n_candidate_miss": 0, "n_unresolved": 0, "n_correct": 0,
                "top1_accuracy": float("nan"), "observation_top1_accuracy": float("nan"),
                "candidate_table_top1_accuracy": float("nan"), "candidate_miss_rate": float("nan")}
    den = outcomes["in_denominator"]
    n = int(den.sum())
    return {
        "n_observations": int(len(outcomes)),
        "n_aligned": n,
        "n_ambiguous_excluded": int((outcomes["alignment_status"] == "ambiguous").sum()),
        "n_unaligned_excluded": int((~den & (outcomes["alignment_status"] != "ambiguous")).sum()),
        "n_candidate_miss": int(outcomes["candidate_miss"].sum()),
        "n_unresolved": int(outcomes["unresolved"].sum()),
        "n_correct": int(outcomes["correct"].sum()),
        "top1_accuracy": float(outcomes["correct"].sum() / n) if n else float("nan"),  # legacy alias
        "observation_top1_accuracy": float(outcomes["correct"].sum() / n) if n else float("nan"),
        "candidate_table_top1_accuracy": (float(outcomes["correct"].sum() / (n - outcomes["candidate_miss"].sum()))
                                          if n - outcomes["candidate_miss"].sum() > 0 else float("nan")),
        "candidate_miss_rate": float(outcomes["candidate_miss"].sum() / n) if n else float("nan"),
    }


# ------------------------------------------------------------- route level
def lcs_length(a: Sequence[int], b: Sequence[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def collapse_repeats(arcs: Sequence[int]) -> list[int]:
    out: list[int] = []
    for a in arcs:
        if not out or out[-1] != a:
            out.append(int(a))
    return out


def reference_route_metrics(pred_arcs: Sequence[int], ref_arcs: Sequence[int],
                            arc_length_m: np.ndarray) -> dict[str, float]:
    """Length-weighted arc-set precision/recall/F1 and sequence similarity vs the reference route.

    set metrics  : over distinct arcs, weighted by arc length (metres)
    lcs_similarity: 2 * LCS(pred, ref) / (len(pred) + len(ref)) on consecutive-collapsed sequences
    (The reference route is supplied reference data, not independently verified ground truth.)
    """
    pred, ref = collapse_repeats(pred_arcs), collapse_repeats(ref_arcs)
    ps, rs = set(pred), set(ref)
    w = lambda s: float(sum(arc_length_m[a] for a in s))
    inter = w(ps & rs)
    precision = inter / w(ps) if w(ps) > 0 else float("nan")
    recall = inter / w(rs) if w(rs) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0 else float("nan"))
    denom = len(pred) + len(ref)
    return {"route_length_precision": precision, "route_length_recall": recall, "route_length_f1": f1,
            "route_lcs_similarity": (2 * lcs_length(pred, ref) / denom) if denom else float("nan")}


def count_matched_discontinuities(states: Sequence[ObservationStates], chosen: Sequence[int],
                                  network: DirectedRoadNetwork, config: DecoderConfig) -> dict[str, float]:
    """Directed-reachability continuity of a chosen state per observation.

    A consecutive pair is *continuous* if the time gap is within gaps.max_gap_seconds and the
    transition between the chosen states has a finite score (directed path exists within the
    search radius and implied speed is plausible). The same test is applied to every pipeline.
    """
    pairs = bad = 0
    reasons: dict[str, int] = {}
    for i in range(len(states) - 1):
        a, b = states[i], states[i + 1]
        pairs += 1
        if b.timestamp - a.timestamp > config.gaps.max_gap_seconds:
            bad += 1
            reasons["large_time_gap"] = reasons.get("large_time_gap", 0) + 1
            continue
        score, _ = calculate_transition_scores(a.select(chosen[i]), b.select(chosen[i + 1]),
                                               network, config.transition)
        if not np.isfinite(score[0, 0]):
            bad += 1
            reasons["no_feasible_transition"] = reasons.get("no_feasible_transition", 0) + 1
    return {"n_consecutive_pairs": pairs, "n_discontinuities": bad,
            "continuity_rate": (1.0 - bad / pairs) if pairs else float("nan"), "reasons": reasons}


def direct_adjacency_breaks(route_segments: Sequence[Sequence[int]], network: DirectedRoadNetwork) -> int:
    """Count consecutive arcs (inside a segment) where to_node(a) != from_node(b)."""
    n = 0
    for seg in route_segments:
        for a, b in zip(seg[:-1], seg[1:]):
            if network.to_node(a) != network.from_node(b):
                n += 1
    return n
