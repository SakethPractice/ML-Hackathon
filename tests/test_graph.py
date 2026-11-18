"""Road graph and geometry tests on small hand-checkable networks."""

import numpy as np
import pandas as pd
import pytest

from pathsnap.graph.geo import (
    LocalProjection,
    angle_diff_deg,
    bearing_deg,
    haversine_m,
    point_segment_distance,
)
from pathsnap.graph.road_graph import RoadGraph


def _graph(nodes, arcs):
    return RoadGraph.from_frames(
        pd.DataFrame(nodes, columns=["lon", "lat"]),
        pd.DataFrame(arcs, columns=["from_node", "to_node"]),
    )


# Four nodes on a line along the equator-ish, plus one isolated node.
#   0 <-> 1 -> 2      3 -> 2       4 (isolated)
NODES = [[8.000, 49.0], [8.001, 49.0], [8.002, 49.0], [8.002, 49.001], [8.5, 49.5]]
ARCS = [[0, 1], [1, 0], [1, 2], [3, 2]]


def test_haversine_known_value():
    # one degree of latitude ~ 111.2 km on the mean sphere
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    # one degree of longitude at 60N is half of that at the equator
    assert haversine_m(0, 60, 1, 60) == pytest.approx(haversine_m(0, 0, 1, 0) / 2, rel=1e-3)


def test_projection_matches_haversine_locally():
    proj = LocalProjection(8.0, 49.0)
    xy = proj.to_xy([8.0, 8.01], [49.0, 49.005])
    flat = np.linalg.norm(xy[1] - xy[0])
    assert flat == pytest.approx(haversine_m(8.0, 49.0, 8.01, 49.005), rel=1e-3)
    back = proj.to_lonlat(xy)
    assert np.allclose(back, [[8.0, 49.0], [8.01, 49.005]])


def test_bearing_and_angle_diff():
    assert bearing_deg(0, 1) == pytest.approx(0)
    assert bearing_deg(1, 0) == pytest.approx(90)
    assert bearing_deg(0, -1) == pytest.approx(180)
    assert angle_diff_deg(350, 10) == pytest.approx(20)
    assert angle_diff_deg(0, 180) == pytest.approx(180)


def test_point_segment_distance_uses_segment_interior():
    a = np.array([[0.0, 0.0]])
    b = np.array([[100.0, 0.0]])
    d, t, _ = point_segment_distance(np.array([[50.0, 10.0]]), a, b)
    # 10 m to the middle of the segment, while both endpoints are ~51 m away
    assert d[0] == pytest.approx(10.0)
    assert t[0] == pytest.approx(0.5)
    d, t, _ = point_segment_distance(np.array([[-30.0, 40.0]]), a, b)
    assert d[0] == pytest.approx(50.0) and t[0] == 0.0


def test_point_segment_distance_zero_length_segment():
    a = np.array([[5.0, 5.0]])
    d, t, _ = point_segment_distance(np.array([[8.0, 9.0]]), a, a.copy())
    assert d[0] == pytest.approx(5.0) and t[0] == 0.0


def test_graph_keeps_ids_and_adjacency():
    g = _graph(NODES, ARCS)
    assert g.n_nodes == 5 and g.n_arcs == 4
    assert g.arc_from.tolist() == [0, 1, 1, 3]
    assert sorted(g.successors(0).tolist()) == [1, 2]  # 0->1 can continue on 1->0 or 1->2
    assert g.successors(2).tolist() == []  # node 2 is a dead end
    assert sorted(g.predecessors(2).tolist()) == [0]
    assert g.follows([0, 2], [2, 0]).tolist() == [True, False]


def test_arc_geometry():
    g = _graph(NODES, ARCS)
    assert g.arc_bearing_deg[2] == pytest.approx(90, abs=1e-6)  # east
    assert g.arc_bearing_deg[1] == pytest.approx(270, abs=1e-6)  # west
    assert g.arc_bearing_deg[3] == pytest.approx(180, abs=1e-6)  # south
    assert g.arc_length_m[0] == pytest.approx(haversine_m(8.0, 49.0, 8.001, 49.0))


def test_connectivity_summary():
    g = _graph(NODES, ARCS)
    s = g.connectivity_summary()
    assert s["n_isolated_nodes"] == 1
    assert s["n_weak_components"] == 1
    # strong: {0,1}, {2}, {3}
    assert s["n_strong_components"] == 3
    assert s["n_dead_end_nodes"] == 1  # node 2
    assert s["n_source_only_nodes"] == 1  # node 3
    assert s["share_arcs_two_way"] == pytest.approx(0.5)


def test_invalid_node_reference_raises():
    with pytest.raises(ValueError):
        _graph(NODES, [[0, 9]])
