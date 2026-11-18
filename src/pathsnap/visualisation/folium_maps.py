"""Folium maps for trips.

``trip_map`` draws, as separate toggleable layers:

* the road network near the GPS track (grey),
* the raw GPS observations (blue dots, start/end markers),
* the reference route if given (green),
* a predicted route if given (red, dashed),
* optional extra point layers (e.g. off-route / excluded fixes).

The same function is meant to be reused for prediction maps later, so it
only takes plain arrays of arc IDs and never reads files itself.
"""

from __future__ import annotations

import folium
import numpy as np

from pathsnap.graph.road_graph import RoadGraph
from pathsnap.graph.spatial_index import ArcIndex

COLOURS = {
    "network": "#9e9e9e",
    "gps": "#1f4fd1",
    "reference": "#1a9e3a",
    "predicted": "#d62728",
    "extra": "#ff8c00",
}


def _unique_undirected(graph: RoadGraph, arcs: np.ndarray) -> np.ndarray:
    """Keep one arc per unordered node pair so two-way roads are drawn once."""
    u = np.minimum(graph.arc_from[arcs], graph.arc_to[arcs])
    v = np.maximum(graph.arc_from[arcs], graph.arc_to[arcs])
    _, first = np.unique(u * graph.n_nodes + v, return_index=True)
    return arcs[np.sort(first)]


def network_arcs_near(graph: RoadGraph, xy: np.ndarray, buffer_m: float, index: ArcIndex | None = None) -> np.ndarray:
    index = index or ArcIndex(graph, step_m=max(25.0, buffer_m / 2))
    # thin the query points; consecutive 1 Hz fixes are a few metres apart
    step = max(1, len(xy) // 2000)
    found = index.arcs_within(xy[::step], buffer_m)
    arcs = np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.int64)
    return arcs


def _route_polylines(graph: RoadGraph, arc_ids: np.ndarray) -> list[list[list[float]]]:
    """Split an arc sequence into connected polylines of [lat, lon] points."""
    lines, cur = [], []
    prev = None
    for a in arc_ids:
        a = int(a)
        f, t = graph.arc_from[a], graph.arc_to[a]
        if prev is None or graph.arc_to[prev] != f:
            if cur:
                lines.append(cur)
            cur = [graph.node_lonlat[f][::-1].tolist()]
        cur.append(graph.node_lonlat[t][::-1].tolist())
        prev = a
    if cur:
        lines.append(cur)
    return lines


def _legend(entries: list[tuple[str, str]]) -> str:
    items = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>'
        for name, c in entries
    )
    return (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px sans-serif">'
        f"{items}</div>"
    )


def trip_map(
    graph: RoadGraph,
    track_lonlat: np.ndarray,
    reference_arcs: np.ndarray | None = None,
    predicted_arcs: np.ndarray | None = None,
    title: str = "",
    network_buffer_m: float = 150.0,
    max_gps_points: int = 3000,
    extra_points: dict[str, np.ndarray] | None = None,
) -> folium.Map:
    track_lonlat = np.asarray(track_lonlat, dtype=float)
    xy = graph.projection.to_xy(track_lonlat[:, 0], track_lonlat[:, 1])

    lat_min, lon_min = track_lonlat[:, 1].min(), track_lonlat[:, 0].min()
    lat_max, lon_max = track_lonlat[:, 1].max(), track_lonlat[:, 0].max()
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
    legend = []

    # road network near the track
    net = folium.FeatureGroup(name="Road network (nearby)", show=True)
    for a in _unique_undirected(graph, network_arcs_near(graph, xy, network_buffer_m)):
        folium.PolyLine(graph.arc_lonlat(a)[:, ::-1].tolist(), color=COLOURS["network"],
                        weight=2, opacity=0.7).add_to(net)
    net.add_to(m)
    legend.append(("Road network", COLOURS["network"]))

    if reference_arcs is not None and len(reference_arcs):
        ref = folium.FeatureGroup(name="Reference route", show=True)
        for line in _route_polylines(graph, np.asarray(reference_arcs)):
            folium.PolyLine(line, color=COLOURS["reference"], weight=5, opacity=0.8).add_to(ref)
        ref.add_to(m)
        legend.append(("Reference route", COLOURS["reference"]))

    if predicted_arcs is not None and len(predicted_arcs):
        pred = folium.FeatureGroup(name="Predicted route", show=True)
        for line in _route_polylines(graph, np.asarray(predicted_arcs)):
            folium.PolyLine(line, color=COLOURS["predicted"], weight=4, opacity=0.9, dash_array="6 6").add_to(pred)
        pred.add_to(m)
        legend.append(("Predicted route", COLOURS["predicted"]))

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(track_lonlat) // max_gps_points)
    for i in range(0, len(track_lonlat), step):
        lon, lat = track_lonlat[i]
        folium.CircleMarker([lat, lon], radius=2, color=COLOURS["gps"], fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"obs {i}").add_to(gps)
    folium.Marker(track_lonlat[0, ::-1].tolist(), tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(gps)
    folium.Marker(track_lonlat[-1, ::-1].tolist(), tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(gps)
    gps.add_to(m)
    gps_label = "GPS observations" + (f" (every {step}th)" if step > 1 else "")
    legend.append((gps_label, COLOURS["gps"]))

    for name, pts in (extra_points or {}).items():
        if len(pts) == 0:
            continue
        grp = folium.FeatureGroup(name=name, show=True)
        for lon, lat in np.asarray(pts)[:: max(1, len(pts) // 1000)]:
            folium.CircleMarker([lat, lon], radius=3, color=COLOURS["extra"], fill=True,
                                fill_opacity=0.9, weight=0).add_to(grp)
        grp.add_to(m)
        legend.append((name, COLOURS["extra"]))

    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;'
            f'padding:4px 8px;border:1px solid #888;font:bold 14px sans-serif">{title}</div>'))
    m.get_root().html.add_child(folium.Element(_legend(legend)))
    return m


def trips_overview_map(points: list[dict]) -> folium.Map:
    """One marker per trip at its start point. ``points``: dicts with trip_id, partition, lon, lat."""
    m = folium.Map(tiles="OpenStreetMap", zoom_start=2, location=[30, 20])
    colours = {"train": "blue", "student_test": "orange"}
    groups = {}
    for p in points:
        part = p["partition"]
        if part not in groups:
            groups[part] = folium.FeatureGroup(name=f"{part} trips")
        folium.CircleMarker([p["lat"], p["lon"]], radius=5, color=colours.get(part, "gray"),
                            fill=True, fill_opacity=0.8,
                            tooltip=f'{part} {p["trip_id"]}').add_to(groups[part])
    for g in groups.values():
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend(
        [(f"{k} trips", v) for k, v in [("train", "#1f4fd1"), ("student_test", "#ff8c00")]])))
    return m
