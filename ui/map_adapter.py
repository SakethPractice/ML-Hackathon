"""Adapt the FastAPI prediction response into a Folium map."""

from collections.abc import Sequence
import math
from typing import Any

import folium


Coordinate = tuple[float, float]


def _validated_coordinates(value: Any, field_name: str) -> list[Coordinate]:
    """Validate an optional array of [latitude, longitude] pairs."""
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"'{field_name}' must be an array of [latitude, longitude] pairs.")

    coordinates: list[Coordinate] = []
    for index, point in enumerate(value):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError(
                f"'{field_name}[{index}]' must contain exactly latitude and longitude."
            )
        latitude, longitude = point
        if (
            isinstance(latitude, bool)
            or isinstance(longitude, bool)
            or not isinstance(latitude, (int, float))
            or not isinstance(longitude, (int, float))
        ):
            raise ValueError(f"'{field_name}[{index}]' coordinates must be numeric.")

        latitude, longitude = float(latitude), float(longitude)
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError(f"'{field_name}[{index}]' coordinates must be finite numbers.")
        if not -90 <= latitude <= 90:
            raise ValueError(f"'{field_name}[{index}]' latitude must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError(f"'{field_name}[{index}]' longitude must be between -180 and 180.")
        coordinates.append((latitude, longitude))
    return coordinates


def parse_prediction_coordinates(
    prediction: Any,
) -> tuple[list[Coordinate], list[list[Coordinate]]]:
    """Read optional GPS observations and segmented ``route_geometry``."""
    if not isinstance(prediction, dict):
        raise ValueError("Prediction response must be a JSON object to display a route map.")
    observations = _validated_coordinates(prediction.get("observations"), "observations")
    raw_segments = prediction.get("route_geometry")
    if raw_segments is None:
        raw_segments = []
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise ValueError("'route_geometry' must be an array of coordinate segments.")
    predicted_segments: list[list[Coordinate]] = []
    for index, segment in enumerate(raw_segments):
        points = _validated_coordinates(segment, f"route_geometry[{index}]")
        if points:
            predicted_segments.append(points)
    return observations, predicted_segments


def create_route_map(
    observations: Sequence[Sequence[float]] | None = None,
    predicted_route: Sequence[Any] | None = None,
    *,
    zoom_start: int = 13,
    observations_label: str = "Original GPS observations",
    predicted_route_label: str = "Predicted route",
) -> folium.Map:
    """Create a map with independently toggleable GPS and prediction layers."""
    gps_points = _validated_coordinates(observations, "observations")
    route_values = list(predicted_route or [])
    if route_values and isinstance(route_values[0], Sequence) and route_values[0] and isinstance(route_values[0][0], Sequence):
        route_segments = [
            _validated_coordinates(segment, f"predicted_route[{index}]")
            for index, segment in enumerate(route_values)
        ]
    else:
        route_segments = [_validated_coordinates(route_values, "predicted_route")] if route_values else []
    route_segments = [segment for segment in route_segments if segment]
    route_points = [point for segment in route_segments for point in segment]
    if not gps_points and not route_points:
        raise ValueError("No valid GPS observations or predicted route coordinates were returned.")

    all_points = gps_points + route_points
    map_view = folium.Map(location=all_points[0], zoom_start=zoom_start, control_scale=True)

    if gps_points:
        gps_layer = folium.FeatureGroup(name=observations_label, show=True)
        if len(gps_points) > 1:
            folium.PolyLine(
                gps_points, color="#2563eb", weight=4, opacity=0.85,
                tooltip=observations_label,
            ).add_to(gps_layer)
        # the line already shows the whole track; dots are thinned for speed
        step = max(1, len(gps_points) // 400)
        for point in gps_points[::step]:
            folium.CircleMarker(
                point, radius=3, color="#2563eb", fill=True, fill_color="#2563eb",
                fill_opacity=0.85,
            ).add_to(gps_layer)
        gps_layer.add_to(map_view)

    if route_points:
        route_layer = folium.FeatureGroup(name=predicted_route_label, show=True)
        for segment in route_segments:
            if len(segment) > 1:
                folium.PolyLine(
                    segment, color="#16803c", weight=5, opacity=0.95,
                    tooltip=predicted_route_label,
                ).add_to(route_layer)
        if route_points:
            folium.Marker(
                route_segments[0][0], tooltip="Predicted route start",
                icon=folium.Icon(color="green", icon="play"),
            ).add_to(route_layer)
        if len(route_points) > 1:
            folium.Marker(
                route_segments[-1][-1], tooltip="Predicted route end",
                icon=folium.Icon(color="darkgreen", icon="flag"),
            ).add_to(route_layer)
        route_layer.add_to(map_view)

    legend = (
        "<div style='position:fixed;top:12px;left:56px;z-index:9999;background:#fff;"
        "border:1px solid #d1d5db;border-radius:8px;padding:8px 12px;font:13px sans-serif;'>"
        f"<div><span style='color:#2563eb'>&#9679;</span> {observations_label}</div>"
        f"<div><span style='color:#16803c'>&#9644;</span> {predicted_route_label}</div></div>"
    )
    map_view.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(map_view)
    if len(all_points) > 1:
        map_view.fit_bounds(all_points)
    return map_view
