"""Per-observation GPS context computed from the track alone.

Nothing here uses the reference route, so the same function runs at training
and inference time.

Columns added (one row per observation, aligned with the track index):

``x``, ``y``             projected position in metres (trip projection)
``dt_prev_s``            time since previous fix (NaN for the first)
``step_prev_m``          haversine distance from the previous fix
``speed_prev_mps``       step_prev_m / dt_prev_s
``heading_deg``          bearing of the movement around this fix, from the
                         fixes ``heading_lag`` steps before and after; NaN when
                         that displacement is shorter than ``min_heading_disp_m``
                         (direction is meaningless while standing still)
``window_disp_m``        straight-line displacement between the fixes
                         ``stationary_window`` steps before and after
``is_stationary``        window_disp_m < stationary_disp_m. GPS jitter makes
                         consecutive-step speed unreliable for a parked vehicle
                         (jitter of several metres per second was seen in the
                         EDA), so a windowed displacement is used instead
``gap_before``           dt_prev_s > gap_s
``segment_id``           increments after every gap; fixes in different segments
                         should not be treated as directly consecutive
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.graph.geo import LocalProjection, bearing_deg, haversine_m


@dataclass(frozen=True)
class TrackContextConfig:
    gap_s: float = 30.0
    heading_lag: int = 2
    min_heading_disp_m: float = 3.0
    stationary_window: int = 15
    stationary_disp_m: float = 30.0

    @classmethod
    def from_dict(cls, d: dict) -> "TrackContextConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _shifted_index(n: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n)
    return np.clip(i - lag, 0, n - 1), np.clip(i + lag, 0, n - 1)


def track_context(track: pd.DataFrame, projection: LocalProjection,
                  config: TrackContextConfig | None = None) -> pd.DataFrame:
    cfg = config or TrackContextConfig()
    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    ts = track["timestamp"].to_numpy(dtype=float)
    n = len(track)
    xy = projection.to_xy(lon, lat)

    dt = np.full(n, np.nan)
    step = np.full(n, np.nan)
    if n > 1:
        dt[1:] = np.diff(ts)
        step[1:] = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)

    lo, hi = _shifted_index(n, cfg.heading_lag)
    d = xy[hi] - xy[lo]
    disp = np.linalg.norm(d, axis=1)
    heading = np.where(disp >= cfg.min_heading_disp_m, bearing_deg(d[:, 0], d[:, 1]), np.nan)

    lo, hi = _shifted_index(n, cfg.stationary_window)
    window_disp = np.linalg.norm(xy[hi] - xy[lo], axis=1)

    gap_before = np.nan_to_num(dt, nan=0.0) > cfg.gap_s
    segment_id = np.cumsum(gap_before)

    return pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "dt_prev_s": dt,
            "step_prev_m": step,
            "speed_prev_mps": speed,
            "heading_deg": heading,
            "window_disp_m": window_disp,
            "is_stationary": window_disp < cfg.stationary_disp_m,
            "gap_before": gap_before,
            "segment_id": segment_id,
        },
        index=track.index,
    )
