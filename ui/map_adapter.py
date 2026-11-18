"""Adapt prediction JSON into a Folium map without changing the ML pipeline.

The provisional API fields are ``observations`` and ``predicted_route``. Both
contain WGS84 ``[latitude, longitude]`` pairs. Keep this response parsing here
so it can be updated independently if the backend contract changes.
"""

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
) -> tuple[list[Coordinate], list[Coordinate]]:
    """Read and validate the provisional coordinate fields from an API response."""
    if not isinstance(prediction, dict):
        raise ValueError("Prediction response must be a JSON object to display a route map.")
    observations = _validated_coordinates(prediction.get("observations"), "observations")
    predicted_route = _validated_coordinates(
        prediction.get("predicted_route"), "predicted_route"
    )
    return observations, predicted_route


def create_route_map(
    observations: Sequence[Sequence[float]] | None = None,
    predicted_route: Sequence[Sequence[float]] | None = None,
    *,
    zoom_start: int = 13,
) -> folium.Map:
    """Create a map with independently toggleable GPS and prediction layers."""
    gps_points = _validated_coordinates(observations, "observations")
    route_points = _validated_coordinates(predicted_route, "predicted_route")
    if not gps_points and not route_points:
        raise ValueError("No valid GPS observations or predicted route coordinates were returned.")

    all_points = gps_points + route_points
    map_view = folium.Map(location=all_points[0], zoom_start=zoom_start, control_scale=True)

    if gps_points:
        gps_layer = folium.FeatureGroup(name="Original GPS observations", show=True)
        if len(gps_points) > 1:
            folium.PolyLine(
                gps_points, color="#2563eb", weight=4, opacity=0.85,
                tooltip="Original GPS observations",
            ).add_to(gps_layer)
        for point in gps_points:
            folium.CircleMarker(
                point, radius=3, color="#2563eb", fill=True, fill_color="#2563eb",
                fill_opacity=0.85,
            ).add_to(gps_layer)
        gps_layer.add_to(map_view)

    if route_points:
        route_layer = folium.FeatureGroup(name="Predicted route", show=True)
        if len(route_points) > 1:
            folium.PolyLine(
                route_points, color="#16803c", weight=5, opacity=0.95,
                tooltip="Predicted route",
            ).add_to(route_layer)
        folium.Marker(
            route_points[0], tooltip="Predicted route start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(route_layer)
        if len(route_points) > 1:
            folium.Marker(
                route_points[-1], tooltip="Predicted route end",
                icon=folium.Icon(color="darkgreen", icon="flag"),
            ).add_to(route_layer)
        route_layer.add_to(map_view)

    folium.LayerControl(collapsed=False).add_to(map_view)
    if len(all_points) > 1:
        map_view.fit_bounds(all_points)
    return map_view
