"""Dataset inventory, data-quality checks, EDA figures and Folium maps.

Partitions read: ``train`` and ``student_test``. The evaluator partition is
not read at all (if the folder exists only that fact is recorded).

Analyses that use the reference route beyond its structure (alignment
distances, label exclusions) are run on ``train`` only.

Usage:  python scripts/run_eda.py
Outputs:
  results/dataset_inventory.csv      one row per trip: files, sizes, row counts, malformed lines
  results/data_quality.csv           one row per trip: validation, extent, connectivity, alignment
  results/missing_values.csv         missing / malformed counts per partition and file type
  results/projection_check.csv       local projection vs haversine arc-length error per trip
  results/eda_partition_summary.csv  partition-level aggregates
  results/figures/eda_*.png
  maps/html/eda_*.html
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates.labels import duplicate_arc_mask, reference_labels
from pathsnap.data.loader import FILE_EXTENSIONS, list_trips, load_trip, trip_dir, trip_files
from pathsnap.data.preprocessing import track_context
from pathsnap.data.validation import validate_trip
from pathsnap.graph.geo import haversine_m
from pathsnap.graph.road_graph import RoadGraph
from pathsnap.visualisation import plots
from pathsnap.visualisation.folium_maps import trip_map, trips_overview_map

PARTITIONS = ["train", "student_test"]
EXPECTED_COLUMNS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}
LABEL_THRESHOLD_M = 50.0


def count_lines(path: Path, n_cols: int) -> tuple[int, int, int]:
    """(rows, blank lines, rows whose token count != n_cols) from the raw text."""
    rows = blank = bad = 0
    with open(path, "rb") as f:
        for line in f:
            tok = line.split()
            if not tok:
                blank += 1
                continue
            rows += 1
            if len(tok) != n_cols:
                bad += 1
    return rows, blank, bad


def inventory_row(partition: str, trip_id: str, data_root: str) -> dict:
    files = trip_files(trip_dir(partition, trip_id, data_root))
    row = {"partition": partition, "trip_id": trip_id, "n_files": len(files)}
    for ext in FILE_EXTENSIONS:
        p = files.get(ext)
        row[f"{ext}_exists"] = p is not None
        if p is None:
            row.update({f"{ext}_bytes": np.nan, f"{ext}_rows": np.nan, f"{ext}_malformed_lines": np.nan,
                        f"{ext}_blank_lines": np.nan})
            continue
        rows, blank, bad = count_lines(p, EXPECTED_COLUMNS[ext])
        row.update({f"{ext}_bytes": p.stat().st_size, f"{ext}_rows": rows,
                    f"{ext}_malformed_lines": bad, f"{ext}_blank_lines": blank})
    return row


def projection_error(graph: RoadGraph) -> dict:
    a, b = graph.arc_xy(np.arange(graph.n_arcs))
    flat = np.linalg.norm(b - a, axis=1)
    ok = graph.arc_length_m > 1.0
    rel = np.abs(flat[ok] - graph.arc_length_m[ok]) / graph.arc_length_m[ok]
    ext = graph.node_xy.max(0) - graph.node_xy.min(0)
    return {
        "network_extent_x_km": ext[0] / 1000, "network_extent_y_km": ext[1] / 1000,
        "arc_len_rel_err_median": float(np.median(rel)), "arc_len_rel_err_max": float(rel.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--results", default="results")
    ap.add_argument("--maps", default="maps/html")
    args = ap.parse_args()
    res = Path(args.results)
    fig_dir = res / "figures"
    map_dir = Path(args.maps)
    res.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    evaluator_present = (Path(args.data_root) / "evaluator_test").is_dir()
    print(f"evaluator_test folder present: {evaluator_present} (not read)")

    inventory, quality, proj_rows = [], [], []
    dts = {p: [] for p in PARTITIONS}
    align_d, align_stat = [], []
    overview = []
    t0 = time.time()

    for part in PARTITIONS:
        for trip_id in list_trips(part, args.data_root):
            inventory.append(inventory_row(part, trip_id, args.data_root))
            trip = load_trip(part, trip_id, args.data_root)
            graph = RoadGraph.from_trip(trip)
            ctx = track_context(trip.track, graph.projection)

            q = validate_trip(trip)
            q.update(graph.connectivity_summary())
            tr = trip.track
            q.update({
                "gps_lon_min": tr.lon.min(), "gps_lon_max": tr.lon.max(),
                "gps_lat_min": tr.lat.min(), "gps_lat_max": tr.lat.max(),
                "gps_lon_mean": tr.lon.mean(), "gps_lat_mean": tr.lat.mean(),
                "gps_path_km": float(np.nansum(ctx["step_prev_m"])) / 1000,
                "gps_extent_x_km": float(np.ptp(ctx["x"])) / 1000,
                "gps_extent_y_km": float(np.ptp(ctx["y"])) / 1000,
                "speed_p50_mps": float(np.nanmedian(ctx["speed_prev_mps"])),
                "speed_p99_mps": float(np.nanpercentile(ctx["speed_prev_mps"], 99)),
                "share_stationary": float(ctx["is_stationary"].mean()),
                "n_segments_after_gap_split": int(ctx["segment_id"].max() + 1),
                "share_arcs_duplicated": float(duplicate_arc_mask(graph).mean()),
            })
            if trip.has_route:
                r = trip.route["arc_id"].to_numpy()
                q["route_length_km"] = float(graph.arc_length_m[r].sum()) / 1000

            if part == "train" and trip.has_route:
                lab = reference_labels(trip, graph, ctx[["x", "y"]].to_numpy(), LABEL_THRESHOLD_M)
                d = lab["align_dist_m"].to_numpy()
                align_d.append(d)
                align_stat.append(ctx["is_stationary"].to_numpy())
                used = np.unique(lab["route_pos"])
                q.update({
                    "align_dist_p50_m": float(np.median(d)), "align_dist_p95_m": float(np.percentile(d, 95)),
                    "align_dist_max_m": float(d.max()),
                    "n_label_excluded": int((~lab["label_valid"]).sum()),
                    "share_label_excluded": float((~lab["label_valid"]).mean()),
                    "share_ref_arc_duplicated": float(lab["ref_has_duplicate"].mean()),
                    "route_share_positions_matched": float(len(used) / len(r)),
                })
            quality.append(q)
            proj_rows.append({"partition": part, "trip_id": trip_id, **projection_error(graph)})
            dts[part].append(np.diff(tr["timestamp"].to_numpy()))
            overview.append({"partition": part, "trip_id": trip_id,
                             "lon": float(tr.lon.iloc[0]), "lat": float(tr.lat.iloc[0])})
            print(f"{part:12s} {trip_id}  {time.time() - t0:6.1f}s", flush=True)

    inv = pd.DataFrame(inventory)
    qual = pd.DataFrame(quality)
    inv.to_csv(res / "dataset_inventory.csv", index=False)
    qual.to_csv(res / "data_quality.csv", index=False, float_format="%.6g")
    pd.DataFrame(proj_rows).to_csv(res / "projection_check.csv", index=False, float_format="%.6g")

    # missing / malformed values per partition and file type
    mv = []
    for part, g in inv.groupby("partition"):
        q = qual[qual.partition == part]
        for ext in FILE_EXTENSIONS:
            mv.append({
                "partition": part, "file": ext,
                "trips_with_file": int(g[f"{ext}_exists"].sum()), "trips_total": len(g),
                "rows": int(g[f"{ext}_rows"].sum()),
                "malformed_lines": int(g[f"{ext}_malformed_lines"].sum()),
                "blank_lines": int(g[f"{ext}_blank_lines"].sum()),
                "missing_values": int(q["n_missing_values"].sum()) if ext == "track" else
                int(q["n_missing_node_values"].sum()) if ext == "nodes" else 0,
                "out_of_range_coords": int(q["n_out_of_range_coords"].sum()) if ext == "track" else
                int(q["n_out_of_range_node_coords"].sum()) if ext == "nodes" else 0,
                "invalid_id_refs": int(q["n_arcs_invalid_node_ref"].sum()) if ext == "arcs" else
                int(q["route_n_invalid_arc_ids"].sum()) if ext == "route" else 0,
            })
    pd.DataFrame(mv).to_csv(res / "missing_values.csv", index=False)

    agg = qual.groupby("partition").agg(
        trips=("trip_id", "count"), obs_total=("n_obs", "sum"), obs_median=("n_obs", "median"),
        obs_min=("n_obs", "min"), obs_max=("n_obs", "max"),
        duration_h_total=("duration_s", lambda s: s.sum() / 3600),
        gps_path_km_total=("gps_path_km", "sum"),
        share_dt_eq_1s_mean=("share_dt_eq_1s", "mean"),
        trips_with_gap_gt_30s=("n_gaps_gt_30s", lambda s: int((s > 0).sum())),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
        route_len_median=("route_len", "median"),
        trips_with_route_break=("route_n_breaks", lambda s: int((s > 0).sum())),
        share_stationary_mean=("share_stationary", "mean"),
    )
    agg.loc["evaluator_test (not supplied)" if not evaluator_present else "evaluator_test (present, not read)"] = np.nan
    agg.to_csv(res / "eda_partition_summary.csv", float_format="%.6g")

    # ------------------------------------------------------------ figures
    plots.obs_per_trip(qual, fig_dir / "eda_obs_per_trip.png")
    plots.sampling_intervals({p: np.concatenate(v) for p, v in dts.items() if v}, fig_dir / "eda_sampling_intervals.png")
    plots.trip_locations(qual, fig_dir / "eda_trip_locations.png")
    plots.network_size(qual, fig_dir / "eda_network_size_connectivity.png")
    if align_d:
        plots.alignment_distances(np.concatenate(align_d), np.concatenate(align_stat), LABEL_THRESHOLD_M,
                                  fig_dir / "eda_train_alignment_distance.png")
    sweep_path = res / "candidate_sweep.csv"
    if sweep_path.exists():
        plots.candidate_sweep(pd.read_csv(sweep_path, dtype={"top_k": str}), fig_dir / "eda_train_candidate_sweep.png")

    # trajectory grid: 6 train + 3 student-test trips spread over the size range
    import matplotlib.pyplot as plt
    picks = []
    for part, n in [("train", 6), ("student_test", 3)]:
        g = qual[qual.partition == part].sort_values("n_obs")
        picks += [(part, t) for t in g["trip_id"].iloc[np.linspace(0, len(g) - 1, n).astype(int)]]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for ax, (part, tid) in zip(axs.flat, picks):
        trip = load_trip(part, tid, args.data_root)
        graph = RoadGraph.from_trip(trip)
        xy = graph.projection.to_xy(trip.track.lon, trip.track.lat)
        plots.draw_trip(ax, graph, xy, trip.route["arc_id"].to_numpy(), title=f"{part} {tid} ({len(xy)} fixes)")
    fig.suptitle("GPS fixes (blue), reference route (green), road network (grey); axes in metres", y=0.995)
    plots._save(fig, fig_dir / "eda_trajectory_grid.png")

    # ------------------------------------------------------------ folium
    trips_overview_map(overview).save(map_dir / "eda_trip_locations.html")
    train_q = qual[qual.partition == "train"]
    typical = train_q.iloc[(train_q["n_obs"] - train_q["n_obs"].median()).abs().argsort().iloc[0]]["trip_id"]
    most_excluded = train_q.sort_values("share_label_excluded").iloc[-1]["trip_id"]
    for part, tid in [("train", typical), ("train", most_excluded),
                      ("student_test", qual[qual.partition == "student_test"]["trip_id"].iloc[0])]:
        trip = load_trip(part, tid, args.data_root)
        graph = RoadGraph.from_trip(trip)
        lonlat = trip.track[["lon", "lat"]].to_numpy()
        extra = None
        if part == "train":
            xy = graph.projection.to_xy(lonlat[:, 0], lonlat[:, 1])
            lab = reference_labels(trip, graph, xy, LABEL_THRESHOLD_M)
            extra = {f"Fixes > {LABEL_THRESHOLD_M:g} m from reference (label excluded)":
                     lonlat[~lab["label_valid"].to_numpy()]}
        m = trip_map(graph, lonlat, reference_arcs=trip.route["arc_id"].to_numpy(),
                     title=f"{part} trip {tid}: GPS, road network, reference route", extra_points=extra)
        m.save(map_dir / f"eda_{part}_{tid}.html")
        print("map", part, tid)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
