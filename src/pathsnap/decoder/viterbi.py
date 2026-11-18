"""Viterbi dynamic programming implemented from scratch (log space).

    delta[t][j] = log_emission[t][j] + max_i ( delta[t-1][i] + log_transition[t-1->t][i][j] )

* Variable candidate counts per position are supported (matrices are n_prev x n_cur).
* ``-inf`` transitions mean "impossible"; they are never accepted.
* If every transition into position t is impossible, or ``break_fn`` requests it, the
  trajectory is *split*: the current segment is backtracked and closed, and a new
  segment starts at t using only emissions. Segments are never silently joined.
* Ties are broken deterministically: ``np.argmax`` returns the lowest index, so
  callers should order states canonically (the decoder sorts by arc id).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class ViterbiSegment:
    positions: list[int]     # indices into the input sequence
    states: list[int]        # chosen candidate index at each position
    log_score: float


@dataclass
class ViterbiResult:
    segments: list[ViterbiSegment]
    breaks: list[tuple[int, int, str]] = field(default_factory=list)  # (prev_pos, pos, reason)
    diagnostics: dict = field(default_factory=dict)


def _backtrack(backptrs: list[np.ndarray], last_state: int) -> list[int]:
    states = [last_state]
    for bp in reversed(backptrs):
        states.append(int(bp[states[-1]]))
    states.reverse()
    return states


def viterbi_decode(log_emissions: Sequence[np.ndarray],
                   transition_fn: Callable[[int, int], np.ndarray],
                   break_fn: Optional[Callable[[int, int], Optional[str]]] = None) -> ViterbiResult:
    """Decode the highest-scoring state sequence.

    ``log_emissions[t]`` is a 1-D array (one entry per candidate at position t).
    ``transition_fn(t-1, t)`` returns the (n_{t-1}, n_t) log-transition matrix.
    ``break_fn(t-1, t)`` may return a reason string to force a segment split.
    """
    n = len(log_emissions)
    if n == 0:
        return ViterbiResult([], [], {"n_positions": 0, "n_segments": 0, "n_breaks": 0,
                                      "break_reasons": {}, "total_log_score": 0.0})
    ems = []
    for t, e in enumerate(log_emissions):
        arr = np.asarray(e, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"position {t}: emissions must be a non-empty 1-D array")
        if np.any(np.isnan(arr)) or np.any(arr == np.inf):
            raise ValueError(f"position {t}: emissions must not contain NaN or +inf")
        if not np.any(np.isfinite(arr)):
            raise ValueError(f"position {t}: no feasible emission candidates")
        ems.append(arr)

    segments: list[ViterbiSegment] = []
    breaks: list[tuple[int, int, str]] = []
    positions = [0]
    delta = ems[0].copy()
    backptrs: list[np.ndarray] = []

    def close_segment() -> None:
        last = int(np.argmax(delta))
        segments.append(ViterbiSegment(list(positions), _backtrack(backptrs, last), float(delta[last])))

    for t in range(1, n):
        reason = break_fn(t - 1, t) if break_fn is not None else None
        new_delta = bp = None
        if reason is None:
            trans = np.asarray(transition_fn(t - 1, t), dtype=float)
            if trans.shape != (ems[t - 1].size, ems[t].size):
                raise ValueError(f"transition matrix for {t - 1}->{t} has shape {trans.shape}, "
                                 f"expected {(ems[t - 1].size, ems[t].size)}")
            if np.any(np.isnan(trans)) or np.any(trans == np.inf):
                raise ValueError(f"transition matrix for {t - 1}->{t} contains NaN or +inf")
            cand = delta[:, None] + trans                       # (n_prev, n_cur)
            bp = np.argmax(cand, axis=0)                        # best predecessor per state
            best = cand[bp, np.arange(cand.shape[1])]
            if np.any(np.isfinite(best)):
                new_delta = best + ems[t]
            else:
                reason = "no_feasible_transition"
        if new_delta is None:
            close_segment()
            breaks.append((t - 1, t, reason or "break"))
            positions, delta, backptrs = [t], ems[t].copy(), []
        else:
            delta = new_delta
            backptrs.append(bp)
            positions.append(t)
    close_segment()

    reasons: dict[str, int] = {}
    for _, _, r in breaks:
        reasons[r] = reasons.get(r, 0) + 1
    return ViterbiResult(segments, breaks, {
        "n_positions": n,
        "n_segments": len(segments),
        "n_breaks": len(breaks),
        "break_reasons": reasons,
        "total_log_score": float(sum(s.log_score for s in segments)),
    })
