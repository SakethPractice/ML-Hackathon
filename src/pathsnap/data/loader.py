"""Readers for the raw PathSnap trip files.

Each trip lives in its own folder ``<partition>/<trip_id>/`` and contains

* ``<trip_id>.track``  longitude, latitude, timestamp (s)   - GPS observations
* ``<trip_id>.nodes``  longitude, latitude                   - row index = node_id
* ``<trip_id>.arcs``   from_node_id, to_node_id              - row index = arc_id
* ``<trip_id>.route``  arc_id per line, in travel order      - reference route

All files are whitespace separated and have no header. The road network is
local to each trip, so node and arc IDs are only meaningful together with the
trip they came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["lon", "lat", "timestamp"]
NODE_COLUMNS = ["lon", "lat"]
ARC_COLUMNS = ["from_node", "to_node"]
ROUTE_COLUMNS = ["arc_id"]

FILE_EXTENSIONS = ("track", "nodes", "arcs", "route")

DEFAULT_DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", "data/raw"))


def _read_table(path: Path, names: list[str], dtype) -> pd.DataFrame:
    """Read a header-less whitespace separated file.

    An empty file gives an empty frame with the expected columns instead of
    raising, so that validation can report it.
    """
    if path.stat().st_size == 0:
        return pd.DataFrame({c: pd.Series(dtype=dtype) for c in names})
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def read_track(path: str | Path) -> pd.DataFrame:
    """GPS observations. ``obs_id`` is the 0-based row position in the file."""
    df = _read_table(Path(path), TRACK_COLUMNS, np.float64)
    df.index.name = "obs_id"
    return df


def read_nodes(path: str | Path) -> pd.DataFrame:
    """Road nodes. The index is the 0-based ``node_id``."""
    df = _read_table(Path(path), NODE_COLUMNS, np.float64)
    df.index.name = "node_id"
    return df


def read_arcs(path: str | Path) -> pd.DataFrame:
    """Directed road arcs. The index is the 0-based ``arc_id``."""
    df = _read_table(Path(path), ARC_COLUMNS, np.int64)
    df.index.name = "arc_id"
    return df


def read_route(path: str | Path) -> pd.DataFrame:
    """Reference route as an ordered list of arc IDs. Index = position in route."""
    df = _read_table(Path(path), ROUTE_COLUMNS, np.int64)
    df.index.name = "route_pos"
    return df


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame | None = None
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def has_route(self) -> bool:
        return self.route is not None


def trip_dir(partition: str, trip_id: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / partition / trip_id


def trip_files(directory: Path) -> dict[str, Path]:
    """Map extension -> path for the files that exist in a trip folder."""
    trip_id = directory.name
    found = {}
    for ext in FILE_EXTENSIONS:
        p = directory / f"{trip_id}.{ext}"
        if p.exists():
            found[ext] = p
    return found


def list_trips(partition: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> list[str]:
    """Sorted trip IDs (folder names) in a partition. Missing partition -> []."""
    root = Path(data_root) / partition
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_trip(
    partition: str,
    trip_id: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    with_route: bool = True,
) -> Trip:
    """Load one trip.

    ``with_route=False`` skips the reference route entirely; the prediction
    path should use that so reference labels never enter inference.
    """
    directory = trip_dir(partition, trip_id, data_root)
    files = trip_files(directory)
    missing = {"track", "nodes", "arcs"} - files.keys()
    if missing:
        raise FileNotFoundError(f"trip {partition}/{trip_id} is missing: {sorted(missing)}")

    route = None
    if with_route and "route" in files:
        route = read_route(files["route"])

    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=read_track(files["track"]),
        nodes=read_nodes(files["nodes"]),
        arcs=read_arcs(files["arcs"]),
        route=route,
        files=files,
    )
