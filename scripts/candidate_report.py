"""Candidate miss rate with the configured generator (configs/data.yaml).

Runs the full generator, including the radius-expansion fallback, on the
training trips and on the student-testing trips. The parameters were chosen
from the training sweep only (scripts/candidate_sweep.py); the student-test
numbers here are a development-stage check and are not used to tune them.

Usage:  python scripts/candidate_report.py
Output: results/candidate_miss_rate.csv (per trip) and a printed summary.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates.labels import candidate_hits, candidate_miss_rate
from pathsnap.candidates.pipeline import load_data_config, prepare_trip
from pathsnap.data.loader import list_trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="results/candidate_miss_rate.csv")
    args = ap.parse_args()
    cfg = load_data_config(args.config)

    rows = []
    t0 = time.time()
    for part in ["train", "student_test"]:
        for trip_id in list_trips(part, cfg.data_root):
            t1 = time.time()
            pt = prepare_trip(part, trip_id, cfg, with_labels=True)
            gen_s = time.time() - t1
            stats = candidate_miss_rate(pt.candidates, pt.labels)
            tab = pt.candidates.table
            hit = candidate_hits(pt.candidates, pt.labels)
            valid = pt.labels["label_valid"].to_numpy()
            moving = ~pt.context["is_stationary"].to_numpy()
            # rank of the reference arc when it is present
            ref = pt.labels["ref_arc_id"].to_numpy()
            ranks = tab.loc[tab["arc_id"].to_numpy() == ref[tab["obs_id"].to_numpy()], "rank"]
            rows.append({
                "partition": part, "trip_id": trip_id, **stats,
                "candidate_miss_rate_moving": float((~hit[valid & moving]).mean()) if (valid & moving).any() else np.nan,
                "share_ref_rank0": float((ranks == 0).sum() / valid.sum()) if valid.any() else np.nan,
                "n_obs_used_fallback": int(tab.loc[tab["radius_used_m"] > cfg.candidates.radius_m, "obs_id"].nunique()),
                "seconds_prepare_trip": gen_s,
            })
            print(f"{part:12s} {trip_id}  miss={stats['candidate_miss_rate']:.4f}  {time.time() - t0:6.1f}s", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6g")

    summary = df.groupby("partition").apply(lambda g: pd.Series({
        "trips": len(g),
        "n_label_valid": g["n_label_valid"].sum(),
        "n_label_excluded": g["n_label_excluded"].sum(),
        "pooled_miss_rate": g["n_miss"].sum() / g["n_label_valid"].sum(),
        "median_trip_miss_rate": g["candidate_miss_rate"].median(),
        "max_trip_miss_rate": g["candidate_miss_rate"].max(),
        "pooled_share_nearest_is_ref": (g["share_ref_rank0"] * g["n_label_valid"]).sum() / g["n_label_valid"].sum(),
        "obs_without_candidates": g["n_obs_without_candidates"].sum(),
        "obs_used_fallback": g["n_obs_used_fallback"].sum(),
        "mean_candidates_per_obs": (g["mean_candidates_per_obs"] * g["n_obs"]).sum() / g["n_obs"].sum(),
    }), include_groups=False)
    summary.to_csv(Path(args.out).with_name("candidate_miss_rate_summary.csv"), float_format="%.6g")
    print(summary.T)


if __name__ == "__main__":
    main()
