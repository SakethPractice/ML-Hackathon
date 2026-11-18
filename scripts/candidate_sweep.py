"""Sweep candidate radius and top-k on the training trips.

For every training trip the reference labels are built once, all arcs within
the largest radius are found once, and each (radius, top_k) combination is
then evaluated by filtering that list. The radius-expansion fallback is
switched off here so the numbers show the effect of the base radius alone;
the share of observations that would need the fallback is reported as
``share_obs_without_candidates``.

Only the training partition is used, so choosing the defaults from this
sweep does not touch student-testing or evaluator data.

Usage:  python scripts/candidate_sweep.py
Output: results/candidate_sweep.csv, results/candidate_sweep_per_trip.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates.generator import CandidateConfig, CandidateGenerator
from pathsnap.candidates.labels import reference_labels
from pathsnap.data.loader import list_trips, load_trip
from pathsnap.data.preprocessing import track_context
from pathsnap.graph.road_graph import RoadGraph

RADII = [15, 25, 35, 50, 75, 100, 150, 200]
TOP_KS = [2, 4, 6, 8, 12, 16, 24, 10_000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-label-dist", type=float, default=50.0)
    args = ap.parse_args()

    big = CandidateConfig(radius_m=max(RADII), top_k=10**9, max_radius_m=max(RADII))
    per_trip = []
    pooled = {(r, k): [0, 0, 0, 0] for r in RADII for k in TOP_KS}  # hits, valid, n_cands, n_empty
    n_obs_total = 0
    t0 = time.time()
    for trip_id in list_trips("train", args.data_root):
        trip = load_trip("train", trip_id, args.data_root)
        graph = RoadGraph.from_trip(trip)
        ctx = track_context(trip.track, graph.projection)
        xy = ctx[["x", "y"]].to_numpy()
        labels = reference_labels(trip, graph, xy, args.max_label_dist)
        cands = CandidateGenerator(graph, big).generate(xy).table

        ref = labels["ref_arc_id"].to_numpy()
        valid = labels["label_valid"].to_numpy()
        c_obs = cands["obs_id"].to_numpy()
        is_ref = cands["arc_id"].to_numpy() == ref[c_obs]
        n = len(labels)
        n_obs_total += n
        for r in RADII:
            for k in TOP_KS:
                keep = (cands["distance_m"].to_numpy() <= r) & (cands["rank"].to_numpy() < k)
                hit = np.zeros(n, dtype=bool)
                hit[c_obs[keep & is_ref]] = True
                has = np.zeros(n, dtype=bool)
                has[c_obs[keep]] = True
                acc = pooled[(r, k)]
                acc[0] += int(hit[valid].sum())
                acc[1] += int(valid.sum())
                acc[2] += int(keep.sum())
                acc[3] += int((~has).sum())
                if k == 8:
                    per_trip.append({
                        "trip_id": trip_id, "radius_m": r, "top_k": k, "n_obs": n,
                        "n_label_valid": int(valid.sum()),
                        "candidate_miss_rate": float(1 - hit[valid].mean()) if valid.any() else np.nan,
                    })
        print(f"{trip_id}  n={n:6d}  valid={valid.mean():.3f}  {time.time() - t0:6.1f}s", flush=True)

    rows = []
    for (r, k), (hits, nv, nc, ne) in pooled.items():
        rows.append({
            "partition": "train", "radius_m": r, "top_k": "all" if k >= 10_000 else k,
            "n_label_valid": nv, "candidate_miss_rate": 1 - hits / nv,
            "mean_candidates_per_obs": nc / n_obs_total,
            "share_obs_without_candidates": ne / n_obs_total,
            "max_label_dist_m": args.max_label_dist,
        })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "candidate_sweep.csv", index=False, float_format="%.6f")
    pd.DataFrame(per_trip).to_csv(out / "candidate_sweep_per_trip.csv", index=False, float_format="%.6f")
    print(pd.DataFrame(rows).pivot(index="radius_m", columns="top_k", values="candidate_miss_rate").round(4))


if __name__ == "__main__":
    main()
