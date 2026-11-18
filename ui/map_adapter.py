"""Map adapter for a route represented as ``(latitude, longitude)`` points.

The backend response schema is not defined yet, so the dashboard deliberately
does not extract coordinates from a prediction response. Once that contract is
agreed, the UI can pass its route points to ``create_route_map``.
"""

from collections.abc import Sequence

import folium


def create_route_map(
    coordinates: Sequence[Sequence[float]], *, zoom_start: int = 13
) -> folium.Map:
    """Create a Folium map from explicit ``(latitude, longitude)`` points."""
    if not coordinates or any(len(point) < 2 for point in coordinates):
        raise ValueError("At least one latitude/longitude pair is required.")
    points = [(float(point[0]), float(point[1])) for point in coordinates]

    map_view = folium.Map(location=points[0], zoom_start=zoom_start, control_scale=True)
    if len(points) > 1:
        folium.PolyLine(points, color="#2563eb", weight=5, opacity=0.9).add_to(map_view)
    folium.Marker(points[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(map_view)
    if len(points) > 1:
        folium.Marker(points[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(map_view)
    return map_view
