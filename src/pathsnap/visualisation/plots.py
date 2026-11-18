"""Matplotlib figures for the EDA. Every function takes data, draws, and saves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from pathsnap.graph.road_graph import RoadGraph  # noqa: E402

PART_COLOURS = {"train": "#1f4fd1", "student_test": "#ff8c00"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def obs_per_trip(quality: pd.DataFrame, path: Path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for part, g in quality.groupby("partition"):
        axs[0].hist(g["n_obs"], bins=np.logspace(2, 4.5, 26), alpha=0.6, label=f"{part} (n={len(g)})",
                    color=PART_COLOURS.get(part))
        axs[1].hist(g["gps_path_km"], bins=np.logspace(-0.5, 2.5, 26), alpha=0.6, label=part,
                    color=PART_COLOURS.get(part))
    axs[0].set(xscale="log", xlabel="GPS observations per trip", ylabel="trips", title="Observations per trip")
    axs[1].set(xscale="log", xlabel="GPS path length (km, sum of steps)", ylabel="trips", title="Trajectory length")
    for ax in axs:
        ax.legend()
    _save(fig, path)


def sampling_intervals(dts: dict[str, np.ndarray], path: Path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.r_[0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 30.5, 60.5, 300.5, 3600.5, 2e5]
    labels = ["1", "2", "3", "4-5", "6-10", "11-30", "31-60", "61-300", "301-3600", ">3600"]
    width = 0.4
    for j, (part, dt) in enumerate(dts.items()):
        h, _ = np.histogram(dt, bins=bins)
        axs[0].bar(np.arange(len(h)) + (j - 0.5) * width, h / h.sum(), width, label=part,
                   color=PART_COLOURS.get(part))
        big = np.sort(dt[dt > 1.5])[::-1]
        axs[1].plot(np.arange(1, len(big) + 1), big, label=part, color=PART_COLOURS.get(part))
    axs[0].set_xticks(range(len(labels)), labels, rotation=45)
    axs[0].set(yscale="log", xlabel="time between consecutive fixes (s)", ylabel="share of steps",
               title="Sampling interval distribution")
    axs[1].set(xscale="log", yscale="log", xlabel="rank", ylabel="gap (s)", title="Gaps > 1 s, sorted")
    for ax in axs:
        ax.legend()
    _save(fig, path)


def trip_locations(quality: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for part, g in quality.groupby("partition"):
        ax.scatter(g["gps_lon_mean"], g["gps_lat_mean"], s=25, label=part, color=PART_COLOURS.get(part),
                   edgecolor="k", linewidth=0.3)
    ax.set(xlabel="longitude (deg)", ylabel="latitude (deg)", title="Mean GPS position of each trip",
           xlim=(-180, 180), ylim=(-60, 80))
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def draw_trip(ax, graph: RoadGraph, xy: np.ndarray, route_arcs=None, pad_m=300, highlight=None, title=""):
    lo = xy.min(0) - pad_m
    hi = xy.max(0) + pad_m
    a, b = graph.arc_xy(np.arange(graph.n_arcs))
    inside = ((a >= lo) & (a <= hi)).all(1) | ((b >= lo) & (b <= hi)).all(1)
    ax.add_collection(LineCollection(np.stack([a[inside], b[inside]], 1), colors="0.75", linewidths=0.5))
    if route_arcs is not None:
        ra, rb = graph.arc_xy(route_arcs)
        ax.add_collection(LineCollection(np.stack([ra, rb], 1), colors="#1a9e3a", linewidths=2.2, alpha=0.8))
    ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color="#1f4fd1", zorder=3)
    if highlight is not None and highlight.any():
        ax.scatter(xy[highlight, 0], xy[highlight, 1], s=3, color="#d62728", zorder=4)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def network_size(quality: pd.DataFrame, path: Path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for part, g in quality.groupby("partition"):
        c = PART_COLOURS.get(part)
        axs[0].scatter(g["n_nodes"], g["n_arcs"], s=15, label=part, color=c)
        axs[1].scatter(g["n_arcs"], g["n_weak_components"], s=15, label=part, color=c)
        axs[2].scatter(g["n_arcs"], g["largest_strong_component_share"], s=15, label=part, color=c)
    axs[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road network size per trip")
    axs[1].set(xscale="log", yscale="log", xlabel="arcs", ylabel="weakly connected components",
               title="Fragmentation")
    axs[2].set(xscale="log", xlabel="arcs", ylabel="share of nodes", title="Largest strongly connected component")
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.3)
    _save(fig, path)


def alignment_distances(dist: np.ndarray, stationary: np.ndarray, threshold: float, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.r_[np.arange(0, 300, 5), 300, 1000]
    ax.hist(dist[~stationary], bins=bins, alpha=0.7, label="moving fixes", color="#1f4fd1")
    ax.hist(dist[stationary], bins=bins, alpha=0.7, label="stationary fixes", color="#ff8c00")
    ax.axvline(threshold, color="k", ls="--", label=f"label threshold {threshold:g} m")
    ax.set(yscale="log", xlim=(0, 320), xlabel="distance from fix to its aligned reference arc (m)",
           ylabel="observations", title="Train: GPS-to-reference-route distance after alignment")
    ax.legend()
    _save(fig, path)


def candidate_sweep(sweep: pd.DataFrame, path: Path):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, g in sweep.groupby("top_k", sort=False):
        g = g.sort_values("radius_m")
        axs[0].plot(g["radius_m"], 100 * g["candidate_miss_rate"], marker="o", ms=3, label=f"top-{k}")
    r50 = sweep[sweep["radius_m"] == 50].copy()
    r50 = r50[r50["top_k"] != "all"].assign(k=lambda d: d["top_k"].astype(int)).sort_values("k")
    axs[1].plot(r50["mean_candidates_per_obs"], 100 * r50["candidate_miss_rate"], marker="o")
    for _, row in r50.iterrows():
        axs[1].annotate(f'k={row["k"]}', (row["mean_candidates_per_obs"], 100 * row["candidate_miss_rate"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    axs[0].set(xlabel="radius (m)", ylabel="candidate miss rate (%)", yscale="symlog",
               title="Train: miss rate vs radius (valid labels)")
    axs[0].set_yscale("symlog", linthresh=0.1)
    axs[0].text(0.98, 0.1, "top-all is 0 for radius >= 50 m by construction:\n"
                "valid labels are within 50 m of their arc",
                transform=axs[0].transAxes, ha="right", fontsize=7, color="0.3")
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].set(xlabel="mean candidates per observation", ylabel="candidate miss rate (%)",
               title="Train: radius 50 m, varying top-k")
    for ax in axs:
        ax.grid(alpha=0.3)
    _save(fig, path)
