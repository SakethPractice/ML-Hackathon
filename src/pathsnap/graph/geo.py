"""Distance and projection helpers.

Coordinates in the raw files are WGS84 longitude/latitude in degrees, so
Euclidean distance on them is not a distance in metres (a degree of
longitude shrinks with cos(latitude)).

Two tools are used:

* ``haversine_m`` - great-circle distance on a sphere of radius 6371008.8 m.
  Used for segment lengths and GPS step lengths.
* ``LocalProjection`` - an equirectangular projection centred on the trip.
  Every trip's road network covers at most a few tens of kilometres, so a
  flat tangent plane is accurate enough for point-to-segment geometry. The
  error against haversine is measured in the EDA (``scripts/run_eda.py``)
  rather than assumed.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean Earth radius


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class LocalProjection:
    """Equirectangular projection to metres around a reference point.

    x points east, y points north, origin at (lon0, lat0).
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = np.radians(1.0) * EARTH_RADIUS_M * np.cos(np.radians(self.lat0))
        self._ky = np.radians(1.0) * EARTH_RADIUS_M

    @classmethod
    def centred_on(cls, lon, lat) -> "LocalProjection":
        """Centre on the middle of the bounding box of the given points."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return cls((lon.min() + lon.max()) / 2, (lat.min() + lat.max()) / 2)

    def to_xy(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return np.column_stack([(lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky])

    def to_lonlat(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([xy[:, 0] / self._kx + self.lon0, xy[:, 1] / self._ky + self.lat0])


def bearing_deg(dx, dy) -> np.ndarray:
    """Compass bearing of a vector in projected metres: 0 = north, 90 = east."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment_distance(p, a, b):
    """Distance from points ``p`` to segments ``a``-``b`` (all (n, 2) arrays in metres).

    Returns ``(distance, t, closest)`` where ``t`` in [0, 1] is the position of
    the closest point along the segment. Zero-length segments fall back to the
    distance to ``a``.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    seg_len2 = np.einsum("ij,ij->i", ab, ab)
    safe = np.where(seg_len2 > 0, seg_len2, 1.0)
    t = np.einsum("ij,ij->i", p - a, ab) / safe
    t = np.where(seg_len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    closest = a + t[:, None] * ab
    dist = np.linalg.norm(p - closest, axis=1)
    return dist, t, closest
